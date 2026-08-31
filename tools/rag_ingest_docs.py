"""ragIngestDocs driver — wires fill.state + fill.ingest + a VectorStoreClient
behind a CLI invoked by the Jenkins `ragIngestDocs` job.

Responsibilities:
  - Resolve `vector_store_id` (CLI arg, env, or already-stored state).
  - Load state from `<platform>/.rag/openai-state.json`.
  - Step 0: build `files_to_process` and `files_removed`.
      * If state.needs_forced_full_scan() → ALL `docs/<type>/en/*.md` files.
      * Else: git-diff `state.last_indexed_docs_commit..HEAD` under `docs/`,
        keeping only the English slice `docs/<type>/en/*.md`, UNION with
        `state.stale_files()` for the changed-files set.
  - Resolve sourceType (the type folder), slug (filename stem), and the logical
    source key "<type>/<stem>.md" (language segment dropped) per file.
  - Run `fill.ingest.ingest_files` (which also adopts untracked VS files
    matching sections it would otherwise upload, and sweeps orphaned VS
    files older than the grace window — disable the latter with --no-sweep).
  - Step ∞ (sentinel stamping, on clean runs only — i.e. no errors):
      * `pipeline_versions` is stamped unconditionally so a version drift
        with no resulting work doesn't leave the sentinel in `None` forever
        (which would force-full-scan every subsequent ingest).
      * `last_indexed_docs_commit` is stamped ONLY when the run did real
        work (sections_uploaded > 0 OR sections_deleted > 0 OR
        files_removed > 0). Stamping on every webhook-triggered no-op
        would create an infinite commit-push-webhook loop: the stamp
        mutates state.json, the Jenkins wrapper commits the change, the
        commit fires the webhook, the next ingest stamps a fresh head
        and recurses. Observed in production: 313 builds in ~90 minutes.
    Errors leave both sentinels untouched so retry happens.
  - Save state. Caller (Jenkins) commits + pushes the state file.

Exit codes:
  0 — success, no ingest errors.
  1 — ingest ran but produced per-file errors (state saved, sentinels NOT
      stamped, affected files marked `stale=True`). Jenkins treats as failure.
  2 — setup error (docs root missing, no vector_store_id, etc.). State not
      modified.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

from fill.chunker import INDEXED_TYPES, SOURCE_TYPES, SourceType
from fill.ingest import DEFAULT_MAX_WORKERS, IngestStats, ingest_files
from fill.openai_client import FakeVectorStoreClient, VectorStoreClient
from fill.state import State, load, save
from fill.versions import pipeline_versions


log = logging.getLogger("rag_ingest_docs")

# Exit codes — used by tests and the Jenkins wrapper.
EXIT_OK = 0
EXIT_INGEST_ERRORS = 1
EXIT_SETUP_ERROR = 2

# Path inside `<platform>` to docs. Layout is TYPE-FIRST: docs/<type>/<lang>/
# (type in language/paradigm/how-to/brief/rules); the RAG corpus is the English
# slice docs/<type>/en/. There is no manifest — the type folder is the source
# of truth. The logical key stays "<type>/<stem>.md" (language segment dropped)
# so the vector store is unaffected by the en/ru<->type/lang layout.
DOCS_SUBDIR = Path("docs")
STATE_RELPATH = Path(".rag/openai-state.json")


# ─── git plumbing (injected for tests) ──────────────────────────────────────


class GitRunner:
    """Real git CLI runner. Tests replace with a fake."""

    def __init__(self, repo: Path) -> None:
        self.repo = repo

    def head_sha(self) -> str:
        return self._run(["rev-parse", "HEAD"]).strip()

    def changed_under(
        self,
        base_sha: str,
        head_sha: str,
        subdir: str,
    ) -> tuple[list[str], list[str]]:
        """Return (added_or_modified, deleted) paths relative to repo root.

        `--no-renames` disables git's rename detection so a `git mv` shows
        up as the old path under D and the new path under A. Without this
        flag, renames collapse to a single R entry containing only the
        destination, and the old source_file would be orphaned in state.
        """
        am = self._run([
            "diff", "--name-only", "--no-renames", "--diff-filter=ACM",
            base_sha, head_sha, "--", subdir,
        ])
        d = self._run([
            "diff", "--name-only", "--no-renames", "--diff-filter=D",
            base_sha, head_sha, "--", subdir,
        ])
        return _splitlines(am), _splitlines(d)

    def _run(self, args: list[str]) -> str:
        out = subprocess.run(
            ["git", *args],
            check=True,
            cwd=self.repo,
            capture_output=True,
            text=True,
        )
        return out.stdout


def _splitlines(s: str) -> list[str]:
    return [line for line in s.splitlines() if line]


# ─── main ───────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Peek at state to learn the persisted vs_id BEFORE building the client.
    # This makes "no --vector-store-id and no env var, but state already
    # knows it" the normal Jenkins flow (Jenkins only sets the env on the
    # first run; subsequent runs read from state).
    state_path = args.platform_root / STATE_RELPATH
    try:
        peek = load(state_path) if state_path.is_file() else State()
    except Exception as e:
        log.error("could not load state %s: %s", state_path, e)
        return EXIT_SETUP_ERROR

    cli_or_env_vs_id = args.vector_store_id or os.environ.get("RAG_VECTOR_STORE_ID") or ""
    resolved_vs_id = cli_or_env_vs_id or peek.vector_store_id or ""
    # The override threaded into run() so first-run dry-runs with no state
    # also persist a usable vs_id. For non-dry-run, only an explicit CLI/env
    # value counts as a deliberate switch.
    override_for_run: str | None = cli_or_env_vs_id or None

    if args.dry_run and not resolved_vs_id:
        # First-run dry-run with no CLI/env/state vs_id: invent a sentinel
        # so the run can complete. The fake client ignores it but it gets
        # persisted into state.json, which keeps the steady-state contract
        # ("vs_id always present after a successful run") consistent.
        resolved_vs_id = "vs_dry-run"
        override_for_run = resolved_vs_id

    if not resolved_vs_id:
        log.error(
            "vector_store_id required on first run: pass --vector-store-id "
            "or set RAG_VECTOR_STORE_ID"
        )
        return EXIT_SETUP_ERROR

    try:
        client = _make_client(args, resolved_vs_id)
    except Exception as e:
        log.error("client setup: %s", e)
        return EXIT_SETUP_ERROR

    git = GitRunner(args.platform_root)
    try:
        exit_code, _ = run(
            platform_root=args.platform_root,
            client=client,
            git=git,
            vector_store_id_override=override_for_run,
            max_workers=args.max_workers,
            sweep=not args.no_sweep,
        )
    except (ValueError, OSError, subprocess.CalledProcessError) as e:
        log.error("setup failure during run: %s", e)
        return EXIT_SETUP_ERROR
    return exit_code


def _positive_int(s: str) -> int:
    """argparse type: parse a positive integer; raise on 0 or negative."""
    n = int(s)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
    return n


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Incremental RAG ingest into OpenAI Vector Store")
    p.add_argument("--platform-root", type=Path, required=True,
                   help="Path to the cloned `platform` repo")
    p.add_argument("--vector-store-id",
                   help="VS id (vs_...). Overrides state; persisted on success. "
                        "Reads $RAG_VECTOR_STORE_ID if omitted.")
    p.add_argument("--dry-run", action="store_true",
                   help="Use FakeVectorStoreClient (no real API calls). "
                        "State is still mutated and saved.")
    p.add_argument("--poll-interval-ms", type=int, default=1000,
                   help="OpenAI VS create_and_poll interval (default 1000ms)")
    p.add_argument("--file-purpose", default="assistants",
                   help='OpenAI Files API "purpose" (default: assistants)')
    p.add_argument("--max-workers", type=_positive_int, default=DEFAULT_MAX_WORKERS,
                   help=f"Parallel section upload/delete pool size "
                        f"(default {DEFAULT_MAX_WORKERS}). Each section upload spends "
                        f"most of its time polling for indexing; the default gets "
                        f"close to Nx speedup before OpenAI rate limits kick in. "
                        f"Set to 1 to debug serially.")
    p.add_argument("--no-sweep", action="store_true",
                   help="Skip the end-of-run orphan sweep: VS files unknown to "
                        "state are left in place (adoption still runs).")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def _make_client(args: argparse.Namespace, vector_store_id: str) -> VectorStoreClient:
    if args.dry_run:
        log.info("--dry-run: using in-memory FakeVectorStoreClient")
        return FakeVectorStoreClient()
    # Lazy import so --dry-run runs without the openai SDK installed.
    from fill.real_openai_client import OpenAIVectorStoreClient
    return OpenAIVectorStoreClient(
        vector_store_id=vector_store_id,
        poll_interval_ms=args.poll_interval_ms,
        file_purpose=args.file_purpose,
    )


# ─── orchestration core (tested directly) ──────────────────────────────────


def run(
    *,
    platform_root: Path,
    client: VectorStoreClient,
    git: GitRunner,
    vector_store_id_override: str | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    sweep: bool = True,
) -> tuple[int, IngestStats]:
    """Run one ingest cycle. Returns (exit_code, stats)."""
    docs_root = platform_root / DOCS_SUBDIR
    state_path = platform_root / STATE_RELPATH

    if not docs_root.is_dir():
        log.error("docs root not found: %s", docs_root)
        return EXIT_SETUP_ERROR, IngestStats()

    state = load(state_path)
    log.info("state: %d files indexed, vs=%s, last_commit=%s",
             len(state.files), state.vector_store_id, state.last_indexed_docs_commit)

    # vs_id resolution: override > state. If override differs from a
    # previously-stored vs_id, the new VS is presumed empty (or unknown),
    # so we wipe state.files and engage the forced-full-scan sentinels.
    # Without this, an empty new VS would never see uploads because the
    # fast-path would skip every file whose recorded file_hash matched.
    if vector_store_id_override:
        if state.vector_store_id and state.vector_store_id != vector_store_id_override:
            log.warning(
                "vector_store_id changing %r → %r; clearing state for full reindex",
                state.vector_store_id, vector_store_id_override,
            )
            state.files = {}
            state.last_indexed_docs_commit = None
            state.pipeline_versions = None
        state.vector_store_id = vector_store_id_override
    if not state.vector_store_id:
        log.error("no vector_store_id in state and none provided")
        return EXIT_SETUP_ERROR, IngestStats()

    current_versions = pipeline_versions()
    source_type_for, slug_for, source_file_for, path_for_key = _make_lookups(docs_root)

    try:
        head_sha = git.head_sha()
        files_to_process, files_removed = _build_file_sets(
            state=state,
            docs_root=docs_root,
            platform_root=platform_root,
            git=git,
            head_sha=head_sha,
            current_versions=current_versions,
            path_for_key=path_for_key,
        )
    except (subprocess.CalledProcessError, OSError) as e:
        log.error("git plumbing failed: %s", e)
        return EXIT_SETUP_ERROR, IngestStats()

    log.info("plan: process %d, remove %d (head=%s)",
             len(files_to_process), len(files_removed), head_sha)

    stats = ingest_files(
        state,
        client,
        files_to_process=files_to_process,
        files_removed=files_removed,
        docs_root=docs_root,
        source_type_for=source_type_for,
        slug_for=slug_for,
        source_file_for=source_file_for,
        max_workers=max_workers,
        sweep=sweep,
    )
    _log_stats(stats)

    # Adopted sections mutate state (new file_ids recorded) exactly like
    # uploads do; swept orphans intentionally do NOT count — the sweep never
    # touches state, and counting it would re-stamp last_commit on runs that
    # changed nothing in the ledger.
    did_real_work = (
        stats.sections_uploaded > 0
        or stats.sections_adopted > 0
        or stats.sections_deleted > 0
        or stats.files_removed > 0
    )

    if stats.errors:
        log.warning("errors present — sentinels NOT stamped; affected files left stale for retry")
    else:
        # `pipeline_versions` stamps unconditionally on clean runs. A bumped
        # version constant produces a `state.needs_forced_full_scan` sentinel
        # trigger, but if every section's payload_hash happens to coincide
        # under both versions (rare but possible) the run finishes with
        # zero uploads — and we still need to advance the stamp, otherwise
        # every subsequent ingest re-enters forced-full-scan.
        state.pipeline_versions = dict(current_versions)

        if did_real_work:
            # `last_indexed_docs_commit` only advances on real work. A no-op
            # ingest that bumped this stamp would mutate state.json, the
            # Jenkins wrapper would commit it, the commit itself would
            # trigger another webhook, and the next ingest would do the
            # same thing → infinite loop. Observed in production: 313
            # builds in ~90 minutes after build #10. Leaving the stamp at
            # the older commit doesn't lose information — the next real
            # docs change will be picked up by `git diff` against that
            # older base just the same.
            state.last_indexed_docs_commit = head_sha
            log.info("stamped end-of-cycle: last_commit=%s, versions=%s",
                     head_sha, current_versions)
        else:
            log.info("no-op ingest — last_commit NOT advanced (versions stamped); avoids webhook loop")

    save(state_path, state)
    log.info("state saved: %s", state_path)

    exit_code = EXIT_OK if not stats.errors else EXIT_INGEST_ERRORS
    return exit_code, stats


# ─── plan: which files to (re)index, which to drop ──────────────────────────


def _build_file_sets(
    *,
    state: State,
    docs_root: Path,
    platform_root: Path,
    git: GitRunner,
    head_sha: str,
    current_versions: dict[str, str],
    path_for_key: Callable[[str], Path],
) -> tuple[list[Path], list[str]]:
    """Step 0: produce `files_to_process` (absolute paths) and `files_removed`
    (logical source_file keys "<type>/<stem>.md")."""
    if state.needs_forced_full_scan(current_versions):
        log.info("forced full scan (no last_commit or pipeline_versions drift)")
        # No base commit to diff against, so removals come from the ledger
        # instead of from git: anything it still tracks that is no longer on
        # disk was deleted or renamed while we were not looking. Without this
        # such a doc keeps its sections in the store forever — the orphan
        # sweep cannot help, since it only considers files the ledger does
        # NOT know, and these it does.
        all_docs = _all_docs(docs_root)
        return all_docs, sorted(set(_tracked_but_gone(state, docs_root, path_for_key))
                                | set(_tracked_but_not_indexed(state)))

    base_sha = state.last_indexed_docs_commit
    if base_sha is None:
        # `needs_forced_full_scan` should have caught this, but defend
        # against future State invariant drift.
        raise RuntimeError("base_sha is None but needs_forced_full_scan was False")
    changed_repo_paths, removed_repo_paths = git.changed_under(
        base_sha, head_sha, str(DOCS_SUBDIR)
    )

    # Only the English doc slice docs/<type>/en/*.md is ingested; everything
    # else under docs/ (ru/, images/, AGENTS.md) is ignored. Crucially, this
    # also ignores the OLD-layout deletions (docs/en/<type>/*.md) produced by
    # the type-first migration commit — their logical keys still exist via the
    # new paths, so dropping them here is what keeps the move churn-free.
    changed_abs = [platform_root / p for p in changed_repo_paths if _is_en_doc(p)]
    removed_keys = [_repo_path_to_key(p) for p in removed_repo_paths if _is_en_doc(p)]

    # Union with stale: state.stale_files() keys are logical source_file keys.
    # `path_for_key` maps them back to docs/<type>/en/<stem>.md. A stale entry
    # whose file no longer exists gets skipped here and removed by a future
    # cycle's git-diff (or by reconcile).
    stale_abs: list[Path] = []
    missing_stale: list[str] = []
    for k in state.stale_files():
        # Stale keys bypass `_is_en_doc` entirely, so a de-indexed branch has to
        # be filtered here as well: without it a failed removal would mark the
        # record stale and the NEXT run would happily re-chunk and re-upload
        # the very article that was just taken out of the index.
        if k.split("/")[0] not in INDEXED_TYPES:
            continue
        path = path_for_key(k)
        if path.is_file():
            stale_abs.append(path)
        else:
            missing_stale.append(k)
    if missing_stale:
        log.debug("stale-but-missing (skipped): %s", missing_stale)

    files_to_process = sorted(set(changed_abs) | set(stale_abs))
    # The diff only sees this window. A doc that went away in an EARLIER one —
    # while the ledger was behind, or during a cycle that failed before
    # stamping — is invisible to it, so the ledger is checked against disk as
    # well. Costs one set difference and no API call.
    return files_to_process, sorted(set(removed_keys)
                                    | set(_tracked_but_gone(state, docs_root, path_for_key))
                                    | set(_tracked_but_not_indexed(state)))


def _tracked_but_not_indexed(state: State) -> list[str]:
    """Ledger keys in a branch that is no longer indexed.

    Nothing else can see these, and it is worth being precise about why. The
    git diff does not: the article did not change, it is still published and
    still on disk. `_tracked_but_gone` does not: it asks whether the FILE is
    missing, and it is not. The orphan sweep does not: it only considers store
    files the ledger does NOT know, and the ledger knows exactly these. Left
    alone they would sit in the store forever — unreachable, since no branch
    filter ever names them again, but still counted, still paid for, and ready
    to be resurrected by any future reconcile that rebuilds the ledger from
    store attributes.

    Idempotent by construction: after one clean run the keys are gone from the
    ledger, so this returns [] for good.
    """
    orphaned = sorted(k for k in state.files if k.split("/")[0] not in INDEXED_TYPES)
    if orphaned:
        log.info("tracked but no longer indexed (%d): %s",
                 len(orphaned), ", ".join(orphaned[:10]))
    return orphaned


def _tracked_but_gone(
    state: State, docs_root: Path, path_for_key: Callable[[str], Path]
) -> list[str]:
    """Ledger keys whose file is no longer on disk — deleted or renamed. The
    orphan sweep cannot see these: it only considers store files the ledger
    does NOT know, and it knows exactly these."""
    gone = sorted(k for k in state.files if not path_for_key(k).is_file())
    if gone:
        log.info("tracked but gone from disk (%d): %s", len(gone), ", ".join(gone[:10]))
    return gone


def _all_docs(docs_root: Path) -> list[Path]:
    # Type-first: ingest the English slice docs/<type>/en/*.md only. `docs/images`
    # and `docs/<type>/AGENTS.md` are skipped (neither sits under <type>/en).
    out: list[Path] = []
    for t in sorted(INDEXED_TYPES):
        en = docs_root / t / "en"
        if en.is_dir():
            out += [p for p in en.rglob("*.md") if p.is_file()]
    return sorted(out)


def _is_en_doc(repo_path: str) -> bool:
    """True for a repo-relative path docs/<type>/en/<file>.md (the ingested slice)."""
    parts = Path(repo_path).parts
    return (
        len(parts) >= 4
        and parts[0] == "docs"
        and parts[1] in INDEXED_TYPES
        and parts[2] == "en"
        and repo_path.endswith(".md")
    )


def _repo_path_to_key(repo_path: str) -> str:
    """docs/<type>/en/<stem>.md -> logical key "<type>/<stem>.md"."""
    parts = Path(repo_path).parts  # ('docs', <type>, 'en', <file>.md)
    return f"{parts[1]}/{parts[3]}"


# ─── folder-based lookups ───────────────────────────────────────────────────


def _make_lookups(docs_root: Path) -> tuple[
    Callable[[Path], SourceType],
    Callable[[Path], str],
    Callable[[Path], str],
    Callable[[str], Path],
]:
    """Resolve, for a doc at docs/<type>/en/<stem>.md:
      - source_type_for → the type folder (`language`/`paradigm`/…)
      - slug_for        → the filename stem
      - source_file_for → the logical key "<type>/<stem>.md" (language segment
        dropped, so the key is stable across the en/ru<->type/lang move and the
        vector store sees no change)
      - path_for_key    → inverse of source_file_for (used to re-find stale files)

    The type folder is the single source of truth (no manifest). A path that is
    not docs/<type>/en/<file>.md raises — surfacing as a per-file setup error in
    `fill.ingest`, never aborting the whole run.
    """
    def slug_for(path: Path) -> str:
        return path.stem

    def source_type_for(path: Path) -> SourceType:
        try:
            rel = path.relative_to(docs_root)
        except ValueError:
            raise ValueError(f"{path} is not under docs root {docs_root}")
        parts = rel.parts
        if len(parts) < 3 or parts[1] != "en":
            raise ValueError(
                f"{rel}: expected <type>/en/<file>.md "
                f"(type in {sorted(SOURCE_TYPES)})"
            )
        folder = parts[0]
        if folder not in SOURCE_TYPES:
            raise ValueError(
                f"{rel}: folder {folder!r} is not a valid sourceType "
                f"{sorted(SOURCE_TYPES)}"
            )
        return folder  # type: ignore[return-value]

    def source_file_for(path: Path) -> str:
        rel = path.relative_to(docs_root)  # <type>/en/<stem>.md
        return f"{rel.parts[0]}/{path.name}"

    def path_for_key(key: str) -> Path:
        t, _, name = key.partition("/")  # "<type>/<stem>.md"
        return docs_root / t / "en" / name

    return source_type_for, slug_for, source_file_for, path_for_key


# ─── logging ────────────────────────────────────────────────────────────────


def _log_stats(stats: IngestStats) -> None:
    log.info(
        "ingest: seen=%d skipped=%d processed=%d removed=%d "
        "uploaded=%d adopted=%d deleted=%d swept=%d sweep_skipped_recent=%d "
        "sweep_skipped_no_created_at=%d errors=%d sweep_errors=%d",
        stats.files_seen,
        stats.files_fast_path_skipped,
        stats.files_processed,
        stats.files_removed,
        stats.sections_uploaded,
        stats.sections_adopted,
        stats.sections_deleted,
        stats.orphans_swept,
        stats.sweep_skipped_recent,
        stats.sweep_skipped_no_created_at,
        len(stats.errors),
        len(stats.sweep_errors),
    )
    for err in stats.errors:
        log.warning("  ! %s", err)
    for err in stats.sweep_errors:
        log.warning("  ~ %s", err)


if __name__ == "__main__":
    sys.exit(main())
