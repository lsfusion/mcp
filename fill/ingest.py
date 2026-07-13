"""fill.ingest — incremental ingest of changed/added/removed docs into the
OpenAI Vector Store, mediated by `state.json`.

This is the Step-1+ orchestrator behind `ragIngestDocs`. Step -1 (state
validation) and Step 0 (build `files_to_process` / `files_removed` from
git-diff ∪ stale ∪ forced-full-scan) are the *caller's* responsibility —
this module just consumes those two sets.

Per-file logic (see RAG-PLAN.md §"ragIngestDocs"):
  • Fast-path: file_hash + pipeline_versions + sourceType all match recorded
    state, and not stale → skip with no chunking or network I/O.
  • Otherwise: chunk the file, diff sections by section_payload_hash:
    – new section_id, or hash changed → upload new file, delete old (if any).
    – section_id present in state but missing in new chunks → delete.
    – section_id unchanged + hash unchanged → leave alone.
  • Removed files (Case B): delete every section's file, drop FileRecord.

Failure handling: any upload/delete error is recorded in `IngestStats.errors`
and the affected file is marked `stale=True` so the next ingest cycle
retries. State always ends in a self-consistent form for the operations
that did succeed — there is no rollback.

Orphans (VS files unknown to state — e.g. a run that uploaded sections but
crashed before its state was persisted, or a replaced section whose delete
failed) are handled by two cooperating mechanisms fed by ONE start-of-run
VS listing:
  • Adoption: a section about to be uploaded whose exact
    `(section_id, section_payload_hash)` already sits in the VS untracked
    gets its file_id recorded instead of re-uploaded — a crashed run's work
    is recovered, not duplicated.
  • Sweep: after processing, untracked files older than the grace window
    are deleted. Sweep failures are soft (`IngestStats.sweep_errors`) — the
    orphan just survives until the next run. `ragRebuildIndex --mode
    reconcile` remains the recovery path for a lost/corrupted state file.
"""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

# Default executor parallelism. Sized for OpenAI's `create_and_poll` profile
# (each section spends ~8s waiting on indexing polls; 8 workers gets us close
# to an 8x speedup before bumping into rate limits). Override via the
# `max_workers` kwarg on `ingest_files`.
DEFAULT_MAX_WORKERS = 8

# Sweep grace window: an untracked VS file younger than this is presumed to
# belong to a concurrent ingest run (its ledger record just isn't visible to
# us) and is left alone. A real orphan has survived at least one full run.
DEFAULT_SWEEP_GRACE_SECONDS = 3600

from fill.chunker import Section, SourceType, chunk_md
from fill.openai_client import AttrValue, VectorStoreClient, VectorStoreSection
from fill.state import FileRecord, SectionRecord, State, mark_stale, remove_file
from fill.versions import pipeline_versions


@dataclass
class IngestStats:
    """Counters + error list for one ingest cycle. Errors are strings —
    the operator inspects them in CI output; programmatic handling is
    intentionally not supported (this isn't a retry orchestrator).

    `sweep_errors` is separate from `errors` on purpose: a failed orphan
    delete (or a failed VS listing) must not fail the run or block sentinel
    stamping — the orphan simply survives until the next run."""

    files_seen: int = 0
    files_fast_path_skipped: int = 0
    files_processed: int = 0
    files_removed: int = 0
    sections_uploaded: int = 0
    sections_adopted: int = 0
    sections_deleted: int = 0
    orphans_swept: int = 0
    sweep_skipped_recent: int = 0
    # Counted apart from `sweep_skipped_recent`: a store that stops reporting
    # created_at would silently disable the sweep, and this counter is the
    # only signal distinguishing that from legitimate grace-window skips.
    sweep_skipped_no_created_at: int = 0
    errors: list[str] = field(default_factory=list)
    sweep_errors: list[str] = field(default_factory=list)


def ingest_files(
    state: State,
    client: VectorStoreClient,
    *,
    files_to_process: list[Path],
    files_removed: list[str],
    docs_root: Path,
    source_type_for: Callable[[Path], SourceType],
    slug_for: Callable[[Path], str],
    source_file_for: Callable[[Path], str] | None = None,
    now: Callable[[], str] | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    sweep: bool = True,
    now_ts: Callable[[], float] | None = None,
) -> IngestStats:
    """Apply one ingest cycle. Mutates `state` in place.

    Args:
        state: loaded from `platform/.rag/openai-state.json`. Caller saves
            the result afterward (this function does NOT save — keeping I/O
            at the caller's boundary makes the unit testable without disk).
        client: VectorStoreClient — real OpenAI wrapper or `FakeVectorStoreClient`.
        files_to_process: absolute paths of `.md` files that may need (re)indexing.
            Already deduplicated by the caller.
        files_removed: source-file paths (relative to `docs_root`) that no
            longer exist; their sections must be removed from the VS.
        docs_root: filesystem root for `source_file` relative-path keys.
        source_type_for / slug_for: resolved from manifest by the caller.
        now: clock injection for tests; defaults to UTC `YYYY-MM-DDTHH:MM:SSZ`.
        max_workers: ThreadPoolExecutor size for parallel upload/delete API
            calls. State mutations are all routed through the main thread
            via `as_completed`, so the only concurrent code path is the
            VectorStoreClient I/O — which is what we want to parallelize
            (OpenAI `create_and_poll` is mostly idle time on the poll loop).
        sweep: delete untracked VS files (orphans) at the end of the run.
            Adoption runs regardless — it only prevents duplicate uploads.
        now_ts: epoch-seconds clock injection for the sweep grace window.
    """
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")
    stats = IngestStats()
    current_versions = pipeline_versions()
    clock = now or _now_utc
    # Default key = path relative to docs_root (legacy/flat callers); type-first
    # callers pass a source_file_for that drops the language segment.
    source_file_for = source_file_for or (lambda p: str(p.relative_to(docs_root)))

    # One VS listing per run feeds both adoption (reuse a crashed run's
    # uploads instead of duplicating them) and the end-of-run orphan sweep.
    # A listing failure degrades gracefully: upload everything, sweep nothing.
    known_ids = {
        srec.file_id
        for rec in state.files.values()
        for srec in rec.sections.values()
    }
    try:
        adoption = _AdoptionIndex(client.list_sections(), known_ids)
    except Exception as e:  # noqa: BLE001 — degrade, don't fail the run
        stats.sweep_errors.append(f"list_sections failed (adoption+sweep skipped): {e}")
        adoption = _AdoptionIndex([], set())
        sweep = False

    # One executor for the whole run. `with` guarantees join even on errors.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        _apply_removals(state, client, files_removed, stats, executor)
        _process_files(
            state, client, files_to_process, docs_root,
            source_type_for, slug_for, source_file_for, clock,
            current_versions, stats, executor, adoption,
        )
        # After _process_files every adoption has been taken, so leftovers()
        # is final. Runs inside the `with` so deletes share the worker pool —
        # the first post-deploy sweep can face a large backlog.
        if sweep:
            _sweep_orphans(
                client, adoption.leftovers(), now_ts or time.time,
                stats, executor,
            )

    return stats


class _AdoptionIndex:
    """VS files unknown to the ledger, indexed for reuse.

    Built from the start-of-run listing. `take` hands an untracked file to a
    section about to be uploaded when `(section_id, section_payload_hash)`
    match: attributes are stamped by `_section_attributes` and payloads are
    byte-identical for equal hashes, so recording the existing file_id is
    equivalent to a fresh upload. Files never taken are orphans;
    `leftovers()` feeds them to the sweep."""

    def __init__(self, listing: list[VectorStoreSection], known_ids: set[str]) -> None:
        self._by_key: dict[tuple[str, str], list[VectorStoreSection]] = {}
        self._unknown: dict[str, VectorStoreSection] = {}
        for vs in listing:
            if vs.file_id in known_ids:
                continue
            self._unknown[vs.file_id] = vs
            sid = vs.attributes.get("section_id")
            payload_hash = vs.attributes.get("section_payload_hash")
            if isinstance(sid, str) and sid and isinstance(payload_hash, str) and payload_hash:
                self._by_key.setdefault((sid, payload_hash), []).append(vs)
        for candidates in self._by_key.values():
            # Lex-smallest file_id wins — same arbitrary-but-stable tiebreak
            # as fill.reconcile._resolve_slots. Losers stay for the sweep.
            candidates.sort(key=lambda v: v.file_id)

    def take(self, section_id: str, payload_hash: str) -> VectorStoreSection | None:
        candidates = self._by_key.get((section_id, payload_hash))
        if not candidates:
            return None
        vs = candidates.pop(0)
        del self._unknown[vs.file_id]
        return vs

    def leftovers(self) -> list[VectorStoreSection]:
        return list(self._unknown.values())


def _sweep_orphans(
    client: VectorStoreClient,
    leftovers: list[VectorStoreSection],
    now_ts: Callable[[], float],
    stats: IngestStats,
    executor: ThreadPoolExecutor,
) -> None:
    """Delete VS files the ledger doesn't know about. Only files older than
    DEFAULT_SWEEP_GRACE_SECONDS go: a concurrent run's fresh uploads are not
    in OUR ledger view, and age is the only signal separating them from real
    orphans. A missing created_at also counts as recent, tracked in its own
    counter (see IngestStats.sweep_skipped_no_created_at)."""
    now = now_ts()
    futures: dict[Future, str] = {}
    for vs in leftovers:
        if vs.created_at is None:
            stats.sweep_skipped_no_created_at += 1
            continue
        if now - vs.created_at < DEFAULT_SWEEP_GRACE_SECONDS:
            stats.sweep_skipped_recent += 1
            continue
        futures[executor.submit(client.delete_section, vs.file_id)] = vs.file_id
    for fut in as_completed(futures):
        file_id = futures[fut]
        try:
            fut.result()
            stats.orphans_swept += 1
        except Exception as e:  # noqa: BLE001 — soft: orphan survives to next run
            stats.sweep_errors.append(f"sweep {file_id}: {e}")


def _process_files(
    state: State,
    client: VectorStoreClient,
    files_to_process: list[Path],
    docs_root: Path,
    source_type_for: Callable[[Path], SourceType],
    slug_for: Callable[[Path], str],
    source_file_for: Callable[[Path], str],
    clock: Callable[[], str],
    current_versions: dict[str, str],
    stats: IngestStats,
    executor: ThreadPoolExecutor,
    adoption: _AdoptionIndex,
) -> None:
    """Per-file loop. Files run sequentially (state mutations per file are
    not partitioned by sid the same way upload/delete are); sections within
    a single file's diff run in parallel through the shared executor."""
    for path in files_to_process:
        stats.files_seen += 1
        # Per-file setup is wrapped: a failing manifest lookup or path-mismatch
        # must not abort the whole run; record the error and move on.
        try:
            source_file = source_file_for(path)
            source_type = source_type_for(path)
            slug = slug_for(path)
        except Exception as e:
            stats.errors.append(f"setup {path}: {e}")
            # Best-effort source_file key for the stale mark.
            try:
                key = source_file_for(path)
            except Exception:
                key = str(path)
            mark_stale(state, key)
            continue
        try:
            raw_bytes = path.read_bytes()
        except OSError as e:
            stats.errors.append(f"read {source_file}: {e}")
            mark_stale(state, source_file)
            continue
        file_hash = _file_hash(raw_bytes)

        if state.can_skip_file_fast_path(source_file, file_hash, current_versions, source_type):
            stats.files_fast_path_skipped += 1
            continue

        try:
            new_sections = chunk_md(path, source_type, slug)
        except Exception as e:
            stats.errors.append(f"chunk {source_file}: {e}")
            mark_stale(state, source_file)
            continue

        _apply_file_diff(
            state=state,
            client=client,
            source_file=source_file,
            source_type=source_type,
            file_hash=file_hash,
            indexed_at=clock(),
            current_versions=current_versions,
            new_sections=new_sections,
            stats=stats,
            executor=executor,
            adoption=adoption,
        )
        stats.files_processed += 1


# ─── case B: removed files ─────────────────────────────────────────────────


def _apply_removals(
    state: State,
    client: VectorStoreClient,
    files_removed: list[str],
    stats: IngestStats,
    executor: ThreadPoolExecutor,
) -> None:
    # Parallel delete across (file, section) pairs of all removed files.
    # All state mutation happens in the main thread after `as_completed`.
    delete_jobs: list[tuple[str, str, str]] = []  # (source_file, sid, file_id)
    for source_file in files_removed:
        rec = state.files.get(source_file)
        if rec is None:
            continue
        for sid, srec in rec.sections.items():
            delete_jobs.append((source_file, sid, srec.file_id))

    futures: dict[Future, tuple[str, str, str]] = {}
    for job in delete_jobs:
        _src, _sid, fid = job
        futures[executor.submit(client.delete_section, fid)] = job

    had_error_per_file: dict[str, bool] = {}
    for fut in as_completed(futures):
        source_file, sid, fid = futures[fut]
        try:
            fut.result()
            stats.sections_deleted += 1
            # Drop from state only after the VS delete succeeds.
            state.files[source_file].sections.pop(sid, None)
        except Exception as e:
            stats.errors.append(f"delete (removed file) {source_file}::{sid}: {e}")
            had_error_per_file[source_file] = True

    # Finalize per-file: keep the FileRecord stale if any of its sections
    # failed to delete; otherwise remove the record entirely.
    for source_file in files_removed:
        rec = state.files.get(source_file)
        if rec is None:
            continue
        if had_error_per_file.get(source_file):
            rec.stale = True
        else:
            remove_file(state, source_file)
            stats.files_removed += 1


# ─── case A / C: per-file section diff ──────────────────────────────────────


def _apply_file_diff(
    *,
    state: State,
    client: VectorStoreClient,
    source_file: str,
    source_type: SourceType,
    file_hash: str,
    indexed_at: str,
    current_versions: dict[str, str],
    new_sections: list[Section],
    stats: IngestStats,
    executor: ThreadPoolExecutor,
    adoption: _AdoptionIndex,
) -> None:
    new_by_id: dict[str, Section] = {s.section_id: s for s in new_sections}
    # `old_rec` is the existing FileRecord, or a throwaway empty one if this
    # is a new file (Case C). The snapshot copy below is what the disappeared-
    # sections diff reads — it must reflect the pre-run state, not the
    # in-progress sections we mutate in the upload loop.
    old_rec = state.files.get(source_file) or FileRecord()
    old_sections = dict(old_rec.sections)
    had_error = False

    # ─── phase 1: adopt or upload new/changed sections ─────────────────────
    delete_old_futures: dict[Future, tuple[str, str]] = {}
    upload_futures: dict[Future, tuple[str, Section, SectionRecord | None]] = {}
    for sid, sec in new_by_id.items():
        old = old_sections.get(sid)
        if old is not None and old.section_payload_hash == sec.section_payload_hash:
            continue  # unchanged
        # A file with this exact payload may already sit in the VS untracked
        # (a previous run uploaded it, then crashed before its ledger record
        # was persisted). Adopt it instead of uploading a duplicate.
        adopted = adoption.take(sid, sec.section_payload_hash)
        if adopted is not None:
            stats.sections_adopted += 1
            _ensure_file_record(state, source_file).sections[sid] = SectionRecord(
                file_id=adopted.file_id,
                section_payload_hash=sec.section_payload_hash,
                heading_path=sec.heading_path,
                source_file=source_file,
            )
            if old is not None:
                d = executor.submit(client.delete_section, old.file_id)
                delete_old_futures[d] = (sid, old.file_id)
            continue
        attributes = _section_attributes(sec, source_file, file_hash)
        filename = _filename_for(sid)
        fut = executor.submit(
            client.upload_section,
            content=sec.payload,
            filename=filename,
            attributes=attributes,
        )
        upload_futures[fut] = (sid, sec, old)

    # State mutation + queueing of the follow-up delete-old happens in the
    # main thread as each upload completes. The old file_id only gets
    # scheduled for deletion AFTER its replacement upload has succeeded —
    # so an upload failure leaves the old file_id intact in the VS.
    for fut in as_completed(upload_futures):
        sid, sec, old = upload_futures[fut]
        try:
            new_fid = fut.result()
            stats.sections_uploaded += 1
        except Exception as e:
            stats.errors.append(f"upload {source_file}::{sid}: {e}")
            had_error = True
            continue

        _ensure_file_record(state, source_file).sections[sid] = SectionRecord(
            file_id=new_fid,
            section_payload_hash=sec.section_payload_hash,
            heading_path=sec.heading_path,
            source_file=source_file,
        )

        if old is not None:
            d = executor.submit(client.delete_section, old.file_id)
            delete_old_futures[d] = (sid, old.file_id)

    # ─── phase 2: parallel delete of replaced + disappeared sections ─────
    # Disappeared = section_ids present in pre-run snapshot but absent
    # from the new chunking. Snapshot reads, not live state, so the upload
    # loop above can't shift the set.
    disappeared_futures: dict[Future, str] = {}
    for sid in old_sections.keys() - new_by_id.keys():
        d = executor.submit(client.delete_section, old_sections[sid].file_id)
        disappeared_futures[d] = sid

    # Wait on replaced-old deletes.
    for fut in as_completed(delete_old_futures):
        sid, old_fid = delete_old_futures[fut]
        try:
            fut.result()
            stats.sections_deleted += 1
        except Exception as e:
            stats.errors.append(
                f"delete-old {source_file}::{sid} (orphan file_id={old_fid}): {e}"
            )
            had_error = True

    # Wait on disappeared deletes.
    for fut in as_completed(disappeared_futures):
        sid = disappeared_futures[fut]
        try:
            fut.result()
            stats.sections_deleted += 1
            state.files[source_file].sections.pop(sid, None)
        except Exception as e:
            stats.errors.append(f"delete {source_file}::{sid}: {e}")
            had_error = True

    # Stamp the FileRecord regardless — fast-path next time depends on it.
    # `stale = had_error` means a clean re-examination clears any prior stale
    # mark (intentional: stale is "please re-examine", not "this file is
    # permanently broken").
    rec = _ensure_file_record(state, source_file)
    rec.file_hash = file_hash
    rec.indexed_with = dict(current_versions)
    rec.indexed_sourceType = source_type
    rec.indexed_at = indexed_at
    rec.stale = had_error


# ─── helpers ───────────────────────────────────────────────────────────────


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _file_hash(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _ensure_file_record(state: State, source_file: str) -> FileRecord:
    rec = state.files.get(source_file)
    if rec is None:
        rec = FileRecord()
        state.files[source_file] = rec
    return rec


def _filename_for(section_id: str) -> str:
    # section_id already starts with the slug (chunker emits
    # `{slug}::{kebab}`), so the slug doesn't need a second appearance in
    # the filename. `::` → `__` for filesystem / OpenAI dashboard friendliness.
    return f"{section_id.replace('::', '__')}.md"


def _section_attributes(
    s: Section, source_file: str, file_hash: str
) -> dict[str, AttrValue]:
    # `indexed_at` lives only in state.FileRecord — keeping it out of the VS
    # attributes makes re-uploads byte-identical (idempotent), so reconcile
    # and future "skip upload if attrs equal" optimizations don't churn.
    return {
        "section_id": s.section_id,
        "slug": s.slug,
        "sourceType": s.source_type,
        "section_name": s.section_name,
        "heading_path": s.heading_path,
        "source_url": s.source_url,
        "section_payload_hash": s.section_payload_hash,
        "source_file": source_file,
        "file_hash": file_hash,
    }
