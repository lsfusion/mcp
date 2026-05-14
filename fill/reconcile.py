"""fill.reconcile — rebuild state.json from the Vector Store.

When local state is lost, corrupted, or suspected of drift, this is the
recovery path. The Vector Store is treated as source of truth: list every
file in it, read its attributes, and rebuild `state.files` from scratch.

This module is intentionally read-only on the VS — it never deletes files,
even ones whose attributes are malformed or whose `(source_file, section_id)`
slot has duplicates. The operator inspects `ReconcileStats` and decides.

After reconcile, the forced-full-scan sentinels are set
(`pipeline_versions=None` and `last_indexed_docs_commit=None`) and every
rebuilt FileRecord is marked `stale=True` with `indexed_with=None`. The
next `ragIngestDocs` therefore visits every doc, recomputes `file_hash`,
and re-stamps `indexed_with` on each file, exiting the sentinel state.

`state.vector_store_id` is preserved (it identifies which VS to talk to).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fill.openai_client import VectorStoreClient, VectorStoreSection
from fill.state import FileRecord, SectionRecord, State, set_reconcile_sentinel


# Attribute keys required on every VS file for it to be absorbed into state.
# Files missing any required attr are reported as `malformed_in_vs` and
# left alone — they likely came from an aborted ingest or a manual upload.
_REQUIRED_STR_ATTRS = (
    "section_id",
    "section_payload_hash",
    "heading_path",
    "source_file",
)


@dataclass
class ReconcileStats:
    """Counters + diagnostic lists for one reconcile run.

    `malformed_in_vs`: VS file_ids whose attributes lack a required key or
        have non-string values. They are NOT absorbed into state and NOT
        deleted from the VS — the operator must clean them manually or
        re-run after deleting them.

    `duplicate_slots`: tuples of (source_file, section_id, losing_file_ids).
        When two VS files claim the same logical slot, the lexicographically
        smallest file_id wins (deterministic, reproducible). Losers are NOT
        deleted from the VS. OpenAI File ids are equal-length `file-<24-char>`
        today, so lex order matches creation order in practice; the tiebreaker
        is arbitrary-but-stable, not "oldest first".
    """

    files_in_vs: int = 0
    malformed_in_vs: list[str] = field(default_factory=list)
    duplicate_slots: list[tuple[str, str, list[str]]] = field(default_factory=list)
    source_files_rebuilt: int = 0
    sections_rebuilt: int = 0


def reconcile(state: State, client: VectorStoreClient) -> ReconcileStats:
    """Rebuild `state.files` from the Vector Store. Mutates `state` in place.

    Steps:
      1. List the VS.
      2. Validate each file's required attributes; bucket malformed.
      3. Group the well-formed files by `source_file`.
      4. For each `source_file`, build a per-section_id slot map. Duplicate
         slots are resolved by smallest-file_id-wins; losers go to stats.
      5. Build a FileRecord per source_file with `stale=True`,
         `indexed_with=None`, `file_hash=None`, and `indexed_sourceType`
         from the lex-smallest-file_id section's attributes (same
         tiebreak rule as duplicate-slot resolution).
      6. Replace `state.files` wholesale.
      7. Set forced-full-scan sentinels.
    """
    stats = ReconcileStats()
    grouped: dict[str, list[VectorStoreSection]] = {}

    # Contract: `client.list_sections()` returns ALL files in the VS, paged
    # through internally. Production wrappers must consume `has_more`/`after`
    # cursors before returning — a partial listing silently corrupts state.
    for vs in client.list_sections():
        stats.files_in_vs += 1
        if not _has_required_attrs(vs.attributes):
            stats.malformed_in_vs.append(vs.file_id)
            continue
        source_file = str(vs.attributes["source_file"])
        grouped.setdefault(source_file, []).append(vs)

    new_files: dict[str, FileRecord] = {}
    for source_file, sections in grouped.items():
        # Deterministic sourceType pick: lex-smallest file_id across the file.
        # All sections of one source_file should share `sourceType`; if they
        # don't, that's a data bug. Note: this is a winner-only check, NOT
        # "scan all sections for the first valid string" — if the winner's
        # sourceType is missing or non-string, we record None and let the
        # next ingest re-stamp it from the manifest.
        winner_attrs = min(sections, key=lambda v: v.file_id).attributes
        st_raw = winner_attrs.get("sourceType")
        indexed_sourcetype = st_raw if isinstance(st_raw, str) else None

        sections_map, duplicates = _resolve_slots(source_file, sections)
        stats.duplicate_slots.extend(duplicates)
        stats.sections_rebuilt += len(sections_map)

        new_files[source_file] = FileRecord(
            file_hash=None,
            indexed_with=None,
            indexed_sourceType=indexed_sourcetype,
            indexed_at=None,
            stale=True,
            sections=sections_map,
        )
        stats.source_files_rebuilt += 1

    state.files = new_files
    set_reconcile_sentinel(state)
    return stats


def _has_required_attrs(attrs: dict) -> bool:
    # Empty strings reject too: an empty `source_file` would group all such
    # files under the "" key, and empty `section_id` would collapse slots.
    for k in _REQUIRED_STR_ATTRS:
        v = attrs.get(k)
        if not isinstance(v, str) or not v:
            return False
    return True


def _resolve_slots(
    source_file: str,
    sections: list[VectorStoreSection],
) -> tuple[dict[str, SectionRecord], list[tuple[str, str, list[str]]]]:
    """Build the section_id → SectionRecord map for one source_file, with
    deterministic tiebreaking on duplicates."""
    # `_has_required_attrs` already verified these are non-empty strings.
    by_sid: dict[str, list[VectorStoreSection]] = {}
    for vs in sections:
        sid: str = vs.attributes["section_id"]  # type: ignore[assignment]
        by_sid.setdefault(sid, []).append(vs)

    out: dict[str, SectionRecord] = {}
    duplicates: list[tuple[str, str, list[str]]] = []
    for sid, candidates in by_sid.items():
        candidates.sort(key=lambda v: v.file_id)
        winner = candidates[0]
        out[sid] = SectionRecord(
            file_id=winner.file_id,
            section_payload_hash=winner.attributes["section_payload_hash"],  # type: ignore[arg-type]
            heading_path=winner.attributes["heading_path"],  # type: ignore[arg-type]
            source_file=source_file,
        )
        if len(candidates) > 1:
            duplicates.append((source_file, sid, [v.file_id for v in candidates[1:]]))
    return out, duplicates
