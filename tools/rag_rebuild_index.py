"""ragRebuildIndex driver — operator-triggered CLI behind the Jenkins
`ragRebuildIndex` job. Wraps `fill.reconcile.reconcile`.

When to run: state.json is suspected stale, lost, corrupted, or out of
sync with what the OpenAI Vector Store actually has. Reconcile reads the
VS authoritatively and rebuilds `state.files` from scratch, marking every
rebuilt file `stale=True` and engaging the forced-full-scan sentinels
(`last_indexed_docs_commit=None`, `pipeline_versions=None`). The next
`ragIngestDocs` cycle re-stamps `file_hash` and `indexed_with` per file.

Unlike `ragIngestDocs`, this driver:
  - Doesn't need the docs/manifest.json (reconcile reads sourceType from
    Vector Store attributes; no per-doc lookup needed).
  - Doesn't consult git — reconcile is unconditional.
  - Has no "partial" exit mode; either it lists the VS and rebuilds, or
    it fails wholesale.

Exit codes:
  0 — success.
  2 — setup error: no vs_id resolvable, VS listing failed, OpenAI auth
      issue, etc. State on disk is not modified unless reconcile itself
      succeeded. Note: a corrupt/unreadable state.json is NOT a setup
      error — recovering from that is exactly what this driver is for.
      A warning is logged and reconcile rebuilds from the VS.

Typical invocation:
  # Operator notices state.json is wrong — rebuild from VS:
  python3 -m tools.rag_rebuild_index --platform-root platform

Note: this driver is NOT the right tool for "begin indexing into a new
(empty) VS". For that, use `ragIngestDocs --vector-store-id vs_NEW`,
which wipes state AND uploads docs. Reconcile against an empty VS would
silently zero out state.files, which is rarely what the operator wants.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from fill.openai_client import FakeVectorStoreClient, VectorStoreClient
from fill.reconcile import ReconcileStats, reconcile
from fill.state import State, load, save


log = logging.getLogger("rag_rebuild_index")

EXIT_OK = 0
EXIT_SETUP_ERROR = 2

STATE_RELPATH = Path(".rag/openai-state.json")


# ─── main ───────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Peek at state to learn the persisted vs_id BEFORE building the client.
    # Recovery semantics: a corrupt or unreadable state.json is the very
    # situation this driver exists to fix, so we proceed with an empty
    # State() and let reconcile rebuild from VS. CLI/env must supply the
    # vs_id in that case (state's stored id is unrecoverable).
    state_path = args.platform_root / STATE_RELPATH
    try:
        peek = load(state_path)
    except Exception as e:
        log.warning("ignoring unreadable state %s (will rebuild from VS): %s", state_path, e)
        peek = State()
        # The corrupt file will be overwritten by save() at the end of run().

    # Resolution order: CLI > state > env. Env is a FIRST-RUN fallback only,
    # not a per-run override — Jenkins typically injects the credential on
    # every build, and treating env as override would silently switch the
    # state's vs_id on every run. Explicit operator override goes via CLI.
    cli_vs_id = args.vector_store_id or ""
    env_vs_id = os.environ.get("RAG_VECTOR_STORE_ID") or ""
    resolved_vs_id = cli_vs_id or peek.vector_store_id or env_vs_id
    # The override that flows into run(): only the CLI flag, or the env
    # fallback when state has nothing yet. Otherwise None → run() leaves
    # state.vector_store_id alone.
    override_for_run: str | None
    if cli_vs_id:
        override_for_run = cli_vs_id
    elif not peek.vector_store_id and env_vs_id:
        override_for_run = env_vs_id
    else:
        override_for_run = None

    if args.dry_run and not resolved_vs_id:
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

    try:
        exit_code, _ = run(
            platform_root=args.platform_root,
            client=client,
            vector_store_id_override=override_for_run,
            preloaded_state=peek,
        )
    except Exception:
        # log.exception preserves the traceback in Jenkins console; this is
        # an operator-recovery job, so failure forensics matter.
        log.exception("reconcile failed")
        return EXIT_SETUP_ERROR
    return exit_code


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rebuild .rag/openai-state.json from the OpenAI Vector Store",
    )
    p.add_argument("--platform-root", type=Path, required=True,
                   help="Path to the cloned `platform` repo")
    p.add_argument("--vector-store-id",
                   help="VS id (vs_...). Overrides any persisted state vs_id "
                        "(the ONLY way to switch). "
                        "If omitted, state's vs_id is used; "
                        "$RAG_VECTOR_STORE_ID is a first-run fallback only.")
    p.add_argument("--dry-run", action="store_true",
                   help="Use FakeVectorStoreClient (no real API calls). "
                        "State is still mutated and saved.")
    p.add_argument("--poll-interval-ms", type=int, default=1000,
                   help="OpenAI VS create_and_poll interval (default 1000ms)")
    p.add_argument("--file-purpose", default="assistants",
                   help='OpenAI Files API "purpose" (default: assistants)')
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def _make_client(args: argparse.Namespace, vector_store_id: str) -> VectorStoreClient:
    if args.dry_run:
        log.info("--dry-run: using in-memory FakeVectorStoreClient")
        return FakeVectorStoreClient()
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
    vector_store_id_override: str | None = None,
    preloaded_state: State | None = None,
) -> tuple[int, ReconcileStats]:
    """Run one reconcile cycle. Returns (exit_code, stats).

    `preloaded_state` lets `main()` pass in an empty State on corrupt-file
    recovery (state.json unreadable → can't be loaded again here). Direct
    callers (tests) pass None and we read from disk.

    Reconcile is wholesale — `state.files` is rebuilt from the Vector Store
    listing. Sentinels (`last_indexed_docs_commit=None`,
    `pipeline_versions=None`) are set so the next `ragIngestDocs` does a
    forced full scan and re-stamps per-file `file_hash` + `indexed_with`.
    `state.vector_store_id` is preserved (or updated by override).
    """
    state_path = platform_root / STATE_RELPATH
    state = preloaded_state if preloaded_state is not None else load(state_path)
    log.info("state before reconcile: %d files, vs=%s, last_commit=%s",
             len(state.files), state.vector_store_id, state.last_indexed_docs_commit)

    # vs_id resolution: override > state. No "switch wipe" here — reconcile
    # itself is a wipe. We just set the id so reconcile knows which VS to
    # consult next time (and the saved state points at the new VS).
    if vector_store_id_override:
        if state.vector_store_id and state.vector_store_id != vector_store_id_override:
            log.info("vector_store_id changing %r → %r (reconcile will rebuild from new VS)",
                     state.vector_store_id, vector_store_id_override)
        state.vector_store_id = vector_store_id_override
    if not state.vector_store_id:
        log.error("no vector_store_id in state and none provided")
        return EXIT_SETUP_ERROR, ReconcileStats()

    log.info("reconciling against vector_store_id=%s", state.vector_store_id)
    stats = reconcile(state, client)
    _log_stats(stats)

    save(state_path, state)
    log.info("state saved: %s", state_path)
    return EXIT_OK, stats


# ─── logging ────────────────────────────────────────────────────────────────


def _log_stats(stats: ReconcileStats) -> None:
    log.info(
        "reconcile: files_in_vs=%d, source_files_rebuilt=%d, sections_rebuilt=%d, "
        "malformed_in_vs=%d, duplicate_slots=%d",
        stats.files_in_vs,
        stats.source_files_rebuilt,
        stats.sections_rebuilt,
        len(stats.malformed_in_vs),
        len(stats.duplicate_slots),
    )
    if stats.malformed_in_vs:
        log.warning("malformed file_ids in VS (manual cleanup needed): %s",
                    stats.malformed_in_vs)
    for src, sid, losers in stats.duplicate_slots:
        log.warning("duplicate slot %s::%s — kept smallest, losers (orphans in VS): %s",
                    src, sid, losers)


if __name__ == "__main__":
    sys.exit(main())
