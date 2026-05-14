"""ragIngestDocs driver — wires fill.state + fill.ingest + a VectorStoreClient
behind a CLI invoked by the Jenkins `ragIngestDocs` job.

Responsibilities:
  - Resolve `vector_store_id` (CLI arg, env, or already-stored state).
  - Load state from `<platform>/.rag/openai-state.json`.
  - Step 0: build `files_to_process` and `files_removed`.
      * If state.needs_forced_full_scan() → ALL `docs/en/*.md` files.
      * Else: git-diff `state.last_indexed_docs_commit..HEAD` under `docs/en/`,
        UNION with `state.stale_files()` for the changed-files set.
  - Resolve sourceType/slug for each file via `docs/manifest.json`.
  - Run `fill.ingest.ingest_files`.
  - Step ∞: ONLY when there are no errors, stamp `pipeline_versions` and
    `last_indexed_docs_commit = HEAD` so the next ingest can fast-path.
    Errors leave the sentinels untouched so retry happens.
  - Save state. Caller (Jenkins) commits + pushes the state file.

Exit codes:
  0 — success, no ingest errors.
  1 — ingest ran but produced per-file errors (state saved, sentinels NOT
      stamped, affected files marked `stale=True`). Jenkins treats as failure.
  2 — setup error (manifest missing, no vector_store_id, etc.). State not
      modified.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

from fill.chunker import SourceType
from fill.ingest import DEFAULT_MAX_WORKERS, IngestStats, ingest_files
from fill.openai_client import FakeVectorStoreClient, VectorStoreClient
from fill.state import State, load, save
from fill.versions import pipeline_versions


log = logging.getLogger("rag_ingest_docs")

# Exit codes — used by tests and the Jenkins wrapper.
EXIT_OK = 0
EXIT_INGEST_ERRORS = 1
EXIT_SETUP_ERROR = 2

# Path inside `<platform>` to docs (English) and to the manifest.
DOCS_SUBDIR = Path("docs/en")
MANIFEST_RELPATH = Path("docs/manifest.json")
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
) -> tuple[int, IngestStats]:
    """Run one ingest cycle. Returns (exit_code, stats)."""
    docs_root = platform_root / DOCS_SUBDIR
    state_path = platform_root / STATE_RELPATH
    manifest_path = platform_root / MANIFEST_RELPATH

    if not docs_root.is_dir():
        log.error("docs root not found: %s", docs_root)
        return EXIT_SETUP_ERROR, IngestStats()
    if not manifest_path.is_file():
        log.error("manifest not found: %s", manifest_path)
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

    try:
        manifest = _load_manifest(manifest_path)
    except (json.JSONDecodeError, OSError, ValueError) as e:
        log.error("manifest load failed: %s", e)
        return EXIT_SETUP_ERROR, IngestStats()
    current_versions = pipeline_versions()

    try:
        head_sha = git.head_sha()
        files_to_process, files_removed = _build_file_sets(
            state=state,
            docs_root=docs_root,
            platform_root=platform_root,
            git=git,
            head_sha=head_sha,
            current_versions=current_versions,
        )
    except (subprocess.CalledProcessError, OSError) as e:
        log.error("git plumbing failed: %s", e)
        return EXIT_SETUP_ERROR, IngestStats()

    log.info("plan: process %d, remove %d (head=%s)",
             len(files_to_process), len(files_removed), head_sha)

    source_type_for, slug_for = _make_lookups(manifest)

    stats = ingest_files(
        state,
        client,
        files_to_process=files_to_process,
        files_removed=files_removed,
        docs_root=docs_root,
        source_type_for=source_type_for,
        slug_for=slug_for,
        max_workers=max_workers,
    )
    _log_stats(stats)

    if not stats.errors:
        state.last_indexed_docs_commit = head_sha
        state.pipeline_versions = dict(current_versions)
        log.info("stamped end-of-cycle: last_commit=%s, versions=%s",
                 head_sha, current_versions)
    else:
        log.warning("errors present — sentinels NOT stamped; affected files left stale for retry")

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
) -> tuple[list[Path], list[str]]:
    """Step 0: produce `files_to_process` (absolute paths) and `files_removed`
    (source_file relative paths, i.e. relative to docs_root)."""
    if state.needs_forced_full_scan(current_versions):
        log.info("forced full scan (no last_commit or pipeline_versions drift)")
        return _all_docs(docs_root), []

    base_sha = state.last_indexed_docs_commit
    if base_sha is None:
        # `needs_forced_full_scan` should have caught this, but defend
        # against future State invariant drift.
        raise RuntimeError("base_sha is None but needs_forced_full_scan was False")
    changed_repo_paths, removed_repo_paths = git.changed_under(
        base_sha, head_sha, str(DOCS_SUBDIR)
    )

    # Map repo-relative paths to absolute paths (for files_to_process)
    # and source_file keys (for files_removed, which is stale-key indexed).
    changed_abs = [platform_root / p for p in changed_repo_paths if p.endswith(".md")]
    removed_keys = [
        str((platform_root / p).relative_to(docs_root))
        for p in removed_repo_paths
        if p.endswith(".md")
    ]

    # Union with stale: state.stale_files() keys are source_file relative
    # paths. A stale entry whose file no longer exists gets skipped here
    # and removed by a future cycle's git-diff (or by reconcile).
    stale_abs: list[Path] = []
    missing_stale: list[str] = []
    for k in state.stale_files():
        path = docs_root / k
        if path.is_file():
            stale_abs.append(path)
        else:
            missing_stale.append(k)
    if missing_stale:
        log.debug("stale-but-missing (skipped): %s", missing_stale)

    files_to_process = sorted(set(changed_abs) | set(stale_abs))
    return files_to_process, removed_keys


def _all_docs(docs_root: Path) -> list[Path]:
    # Flat convention: docs/en/ contains *.md files directly, no nesting.
    # If that ever changes, both this and `GitRunner.changed_under`'s
    # path-filter need updating; right now they're symmetric on `*.md`
    # under docs_root itself.
    return sorted(p for p in docs_root.glob("*.md") if p.is_file())


# ─── manifest lookup ───────────────────────────────────────────────────────


def _load_manifest(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object, got {type(data).__name__}")
    return data


def _make_lookups(manifest: dict[str, dict]) -> tuple[
    Callable[[Path], SourceType],
    Callable[[Path], str],
]:
    def slug_for(path: Path) -> str:
        return path.stem

    def source_type_for(path: Path) -> SourceType:
        slug = path.stem
        entry = manifest.get(slug)
        if entry is None:
            raise KeyError(f"slug {slug!r} not in manifest")
        st = entry.get("sourceType")
        if not isinstance(st, str):
            raise ValueError(f"manifest[{slug!r}].sourceType missing or non-string")
        return st  # type: ignore[return-value]

    return source_type_for, slug_for


# ─── logging ────────────────────────────────────────────────────────────────


def _log_stats(stats: IngestStats) -> None:
    log.info(
        "ingest: seen=%d skipped=%d processed=%d removed=%d "
        "uploaded=%d deleted=%d errors=%d",
        stats.files_seen,
        stats.files_fast_path_skipped,
        stats.files_processed,
        stats.files_removed,
        stats.sections_uploaded,
        stats.sections_deleted,
        len(stats.errors),
    )
    for err in stats.errors:
        log.warning("  ! %s", err)


if __name__ == "__main__":
    sys.exit(main())
