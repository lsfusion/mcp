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
that did succeed — there is no rollback. Orphans (new file uploaded but
state still pointing at old, etc.) are caught by
`ragRebuildIndex --mode reconcile`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from fill.chunker import Section, SourceType, chunk_md
from fill.openai_client import AttrValue, VectorStoreClient
from fill.state import FileRecord, SectionRecord, State, mark_stale, remove_file
from fill.versions import pipeline_versions


@dataclass
class IngestStats:
    """Counters + error list for one ingest cycle. Errors are strings —
    the operator inspects them in CI output; programmatic handling is
    intentionally not supported (this isn't a retry orchestrator)."""

    files_seen: int = 0
    files_fast_path_skipped: int = 0
    files_processed: int = 0
    files_removed: int = 0
    sections_uploaded: int = 0
    sections_deleted: int = 0
    errors: list[str] = field(default_factory=list)


def ingest_files(
    state: State,
    client: VectorStoreClient,
    *,
    files_to_process: list[Path],
    files_removed: list[str],
    docs_root: Path,
    source_type_for: Callable[[Path], SourceType],
    slug_for: Callable[[Path], str],
    now: Callable[[], str] | None = None,
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
    """
    stats = IngestStats()
    current_versions = pipeline_versions()
    clock = now or _now_utc

    _apply_removals(state, client, files_removed, stats)

    for path in files_to_process:
        stats.files_seen += 1
        # Per-file setup is wrapped: a failing manifest lookup or path-mismatch
        # must not abort the whole run; record the error and move on.
        try:
            source_file = str(path.relative_to(docs_root))
            source_type = source_type_for(path)
            slug = slug_for(path)
        except Exception as e:
            stats.errors.append(f"setup {path}: {e}")
            # Best-effort source_file key for the stale mark.
            try:
                key = str(path.relative_to(docs_root))
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
            slug=slug,
            file_hash=file_hash,
            indexed_at=clock(),
            current_versions=current_versions,
            new_sections=new_sections,
            stats=stats,
        )
        stats.files_processed += 1

    return stats


# ─── case B: removed files ─────────────────────────────────────────────────


def _apply_removals(
    state: State,
    client: VectorStoreClient,
    files_removed: list[str],
    stats: IngestStats,
) -> None:
    for source_file in files_removed:
        rec = state.files.get(source_file)
        if rec is None:
            continue
        had_error = False
        for sid in list(rec.sections.keys()):
            srec = rec.sections[sid]
            try:
                client.delete_section(srec.file_id)
                stats.sections_deleted += 1
                # Drop from state only after the VS delete succeeds — order
                # matters; a future refactor that moves the `del` above must
                # be rejected.
                del rec.sections[sid]
            except Exception as e:
                stats.errors.append(f"delete (removed file) {source_file}::{sid}: {e}")
                had_error = True
        if had_error:
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
    slug: str,
    file_hash: str,
    indexed_at: str,
    current_versions: dict[str, str],
    new_sections: list[Section],
    stats: IngestStats,
) -> None:
    new_by_id: dict[str, Section] = {s.section_id: s for s in new_sections}
    # `old_rec` is the existing FileRecord, or a throwaway empty one if this
    # is a new file (Case C). The snapshot copy below is what the disappeared-
    # sections diff reads — it must reflect the pre-run state, not the
    # in-progress sections we mutate in the upload loop.
    old_rec = state.files.get(source_file) or FileRecord()
    old_sections = dict(old_rec.sections)
    had_error = False

    # Replace or insert each new section.
    for sid, sec in new_by_id.items():
        old = old_sections.get(sid)
        if old is not None and old.section_payload_hash == sec.section_payload_hash:
            continue  # unchanged
        attributes = _section_attributes(sec, source_file, file_hash)
        filename = _filename_for(slug, sid)
        try:
            new_fid = client.upload_section(
                content=sec.payload, filename=filename, attributes=attributes
            )
            stats.sections_uploaded += 1
        except Exception as e:
            stats.errors.append(f"upload {source_file}::{sid}: {e}")
            had_error = True
            continue

        # State now points at the new file_id; old becomes a reconcile orphan
        # if the delete below fails.
        _ensure_file_record(state, source_file).sections[sid] = SectionRecord(
            file_id=new_fid,
            section_payload_hash=sec.section_payload_hash,
            heading_path=sec.heading_path,
            source_file=source_file,
        )

        if old is not None:
            try:
                client.delete_section(old.file_id)
                stats.sections_deleted += 1
            except Exception as e:
                # Include old.file_id so an operator can clean it manually if
                # reconcile hasn't yet run.
                stats.errors.append(
                    f"delete-old {source_file}::{sid} (orphan file_id={old.file_id}): {e}"
                )
                had_error = True

    # Delete sections that disappeared from the new chunking. The set
    # difference is computed against the pre-run snapshot above, not the
    # mutated `state.files[source_file].sections`.
    for sid in old_sections.keys() - new_by_id.keys():
        try:
            client.delete_section(old_sections[sid].file_id)
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


def _filename_for(slug: str, section_id: str) -> str:
    # Avoid `::` in filenames — some filesystems and the OpenAI dashboard
    # render double-colons awkwardly. Underscores round-trip cleanly.
    return f"{slug}--{section_id.replace('::', '__')}.md"


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
