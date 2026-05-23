"""fill.state — typed read/write for platform/.rag/openai-state.json.

State is the local cache of "what we've uploaded to the Vector Store" —
NOT the source of truth (OpenAI attributes are). It's a fast lookup that
lets `ragIngestDocs` skip files whose content + pipeline versions +
sourceType haven't changed (the fast-path), and lets `ragRebuildIndex
--mode reconcile` recover when state is lost or out of sync.

See RAG-PLAN.md §"Contracts / openai-state.json" for the schema.

Atomic-write contract: save() writes to <path>.tmp then atomically renames
to <path>. A crash mid-write leaves either the prior state or no .tmp file
(harmless leftover) — never a partially-written file.

Sentinel semantics:
  - `state.last_indexed_docs_commit is None` → forced-full-scan trigger
    for the next ragIngestDocs (set by reconcile).
  - `state.pipeline_versions is None` → forced-full-scan trigger (same source).
  - `file.indexed_with is None` → unknown/unreconciled; treated as
    mismatch by fast-path skip.
  - `file.stale is True` → the file goes into `files_to_process` in
    Step 0 regardless of docs git-diff.
"""

from __future__ import annotations

import json
import os
import stat as _stat
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# Soft cap on state.json size to bound CI memory in case of corruption.
# Real state for ~350 files × ~5 sections × ~300 bytes per record ≈ 500 KB;
# 10 MB allows ~20x growth without hitting the cap.
MAX_STATE_BYTES = 10 * 1024 * 1024


# ─── dataclasses ────────────────────────────────────────────────────────────


@dataclass
class SectionRecord:
    """One section's per-OpenAI-file identity. Stored under
    `state.files[<source_file>].sections[<section_id>]`."""

    file_id: str                 # OpenAI File ID (file-...)
    section_payload_hash: str    # sha256:... — primary reindex signal
    heading_path: str            # human-readable breadcrumb (also in OpenAI attrs)
    source_file: str             # parent .md path, for reconcile flatten


@dataclass
class FileRecord:
    """One source `.md` file's state."""

    file_hash: str | None = None
    indexed_with: dict[str, str] | None = None   # None = unknown (reconcile sentinel)
    indexed_sourceType: str | None = None
    indexed_at: str | None = None                # ISO UTC `YYYY-MM-DDTHH:MM:SSZ`
    stale: bool = False
    sections: dict[str, SectionRecord] = field(default_factory=dict)


@dataclass
class State:
    """Root state object. Schema-aligned with openai-state.json."""

    vector_store_id: str | None = None
    last_indexed_docs_commit: str | None = None        # None = forced-full-scan sentinel
    pipeline_versions: dict[str, str] | None = None    # None = forced-full-scan sentinel
    files: dict[str, FileRecord] = field(default_factory=dict)

    # ─── fast-path / drift ──────────────────────────────────────────────────

    def needs_forced_full_scan(self, current_versions: dict[str, str]) -> bool:
        """Return True if the next ragIngestDocs must scan all docs/<type>/en/ files
        instead of using git-diff. Triggers: sentinel after reconcile, or
        pipeline_versions drift from `fill.versions`.
        """
        if self.last_indexed_docs_commit is None:
            return True
        if self.pipeline_versions is None:
            return True
        if self.pipeline_versions != current_versions:
            return True
        return False

    def stale_files(self) -> set[str]:
        """Source files flagged stale by Step -1 reconciliation. Step 0
        unions these with git-diff to form `files_to_process`.
        """
        return {path for path, rec in self.files.items() if rec.stale}

    def can_skip_file_fast_path(
        self,
        source_file: str,
        current_file_hash: str,
        current_versions: dict[str, str],
        current_source_type: str,
    ) -> bool:
        """Fast-path skip predicate for ragIngestDocs case A.

        Returns True if all four match the recorded state — file content,
        pipeline versions, sourceType, and NOT stale.
        """
        rec = self.files.get(source_file)
        if rec is None:
            return False
        if rec.stale:
            return False
        if rec.file_hash != current_file_hash:
            return False
        if rec.indexed_with is None or rec.indexed_with != current_versions:
            return False
        if rec.indexed_sourceType != current_source_type:
            return False
        return True


# ─── load ───────────────────────────────────────────────────────────────────


def load(path: Path) -> State:
    """Read state.json. Missing file → new empty State. Malformed JSON or
    wrong shape raises ValueError (callers should treat as a CI failure,
    not auto-recovery; `ragRebuildIndex --mode reconcile` is the operator's
    tool for state recovery).
    """
    if not path.exists():
        return State()
    try:
        st = path.stat()
    except OSError as e:
        raise ValueError(f"could not stat {path}: {e}") from e
    if not _stat.S_ISREG(st.st_mode):
        raise ValueError(f"{path}: not a regular file (mode={oct(st.st_mode)})")
    if st.st_size > MAX_STATE_BYTES:
        raise ValueError(
            f"{path}: file size {st.st_size} bytes exceeds cap {MAX_STATE_BYTES} bytes"
        )
    try:
        raw_text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as e:
        raise ValueError(f"could not read {path}: {e}") from e
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: invalid JSON: {e}") from e
    except RecursionError as e:
        raise ValueError(f"{path}: JSON nesting too deep") from e
    return _from_dict(raw)


def _from_dict(raw: Any) -> State:
    if not isinstance(raw, dict):
        raise ValueError(f"state must be a JSON object, got {type(raw).__name__}")

    files_raw = raw.get("files", {})
    if not isinstance(files_raw, dict):
        raise ValueError(f"state.files must be an object, got {type(files_raw).__name__}")

    files: dict[str, FileRecord] = {}
    for src, frec_raw in files_raw.items():
        if not isinstance(frec_raw, dict):
            raise ValueError(f"state.files[{src!r}] must be an object")
        sections_raw = frec_raw.get("sections", {})
        if not isinstance(sections_raw, dict):
            raise ValueError(f"state.files[{src!r}].sections must be an object")
        sections: dict[str, SectionRecord] = {}
        for sid, srec_raw in sections_raw.items():
            if not isinstance(srec_raw, dict):
                raise ValueError(f"state.files[{src!r}].sections[{sid!r}] must be an object")
            for k in ("file_id", "section_payload_hash", "heading_path", "source_file"):
                if k not in srec_raw or not isinstance(srec_raw[k], str):
                    raise ValueError(
                        f"state.files[{src!r}].sections[{sid!r}] missing or non-string {k!r}"
                    )
            sections[sid] = SectionRecord(
                file_id=srec_raw["file_id"],
                section_payload_hash=srec_raw["section_payload_hash"],
                heading_path=srec_raw["heading_path"],
                source_file=srec_raw["source_file"],
            )

        indexed_with = _validated_versions_map(
            frec_raw.get("indexed_with"), f"state.files[{src!r}].indexed_with"
        )

        stale_raw = frec_raw.get("stale", False)
        if not isinstance(stale_raw, bool):
            raise ValueError(
                f"state.files[{src!r}].stale must be bool, got {type(stale_raw).__name__}"
            )

        files[src] = FileRecord(
            file_hash=_opt_str(frec_raw.get("file_hash"), f"state.files[{src!r}].file_hash"),
            indexed_with=indexed_with,
            indexed_sourceType=_opt_str(
                frec_raw.get("indexed_sourceType"),
                f"state.files[{src!r}].indexed_sourceType",
            ),
            indexed_at=_opt_str(frec_raw.get("indexed_at"), f"state.files[{src!r}].indexed_at"),
            stale=stale_raw,
            sections=sections,
        )

    pipeline_versions = _validated_versions_map(
        raw.get("pipeline_versions"), "state.pipeline_versions"
    )

    return State(
        vector_store_id=_opt_str(raw.get("vector_store_id"), "state.vector_store_id"),
        last_indexed_docs_commit=_opt_str(
            raw.get("last_indexed_docs_commit"), "state.last_indexed_docs_commit"
        ),
        pipeline_versions=pipeline_versions,
        files=files,
    )


def _validated_versions_map(value: Any, key_label: str) -> dict[str, str] | None:
    """Validate a `dict[str, str] | None` map (used by `pipeline_versions`
    and `FileRecord.indexed_with`). Returns a fresh dict (defensive copy)
    so callers don't share references with the parsed JSON tree."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{key_label} must be object or null, got {type(value).__name__}")
    out: dict[str, str] = {}
    for k, v in value.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError(
                f"{key_label}[{k!r}] must be string→string, got {type(v).__name__}"
            )
        out[k] = v
    return out


def _opt_str(value: Any, key_label: str) -> str | None:
    """None passes through; everything else must be str."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key_label} must be string or null, got {type(value).__name__}")
    return value


# ─── save ───────────────────────────────────────────────────────────────────


def save(path: Path, state: State) -> None:
    """Atomic write: serialize to <path>.tmp + flush + fsync + os.replace().

    A crash mid-write leaves either:
      - the prior state.json intact (if rename hasn't happened), or
      - a leftover .tmp file (cleaned on next run via tmp file overwrite).
    Never a partially-written state.json.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    payload = json.dumps(_to_dict(state), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        # BaseException catches KeyboardInterrupt/SystemExit so Ctrl-C mid-write
        # still cleans the .tmp (prior state.json stays intact). `raise` re-raises.
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def _to_dict(state: State) -> dict:
    return {
        "vector_store_id": state.vector_store_id,
        "last_indexed_docs_commit": state.last_indexed_docs_commit,
        "pipeline_versions": state.pipeline_versions,
        "files": {
            src: {
                "file_hash": rec.file_hash,
                "indexed_with": rec.indexed_with,
                "indexed_sourceType": rec.indexed_sourceType,
                "indexed_at": rec.indexed_at,
                "stale": rec.stale,
                "sections": {sid: asdict(srec) for sid, srec in rec.sections.items()},
            }
            for src, rec in state.files.items()
        },
    }


# ─── mutators ───────────────────────────────────────────────────────────────


def mark_stale(state: State, source_file: str) -> None:
    """Mark a file as needing reprocessing on next ragIngestDocs (Step 0
    `stale_files` set). Creates a stub FileRecord if the file isn't tracked.
    """
    rec = state.files.get(source_file)
    if rec is None:
        rec = FileRecord(stale=True)
        state.files[source_file] = rec
    else:
        rec.stale = True


def set_reconcile_sentinel(state: State) -> None:
    """Set forced-full-scan sentinels (called by ragRebuildIndex reconcile):
    pipeline_versions=None AND last_indexed_docs_commit=None.
    Together they guarantee Step 0 ignores git-diff and processes every file.
    """
    state.pipeline_versions = None
    state.last_indexed_docs_commit = None


def upsert_section(
    state: State,
    source_file: str,
    section_id: str,
    record: SectionRecord,
) -> None:
    """Add or replace one section's record under its source_file."""
    if source_file not in state.files:
        state.files[source_file] = FileRecord()
    state.files[source_file].sections[section_id] = record


def remove_section(state: State, source_file: str, section_id: str) -> None:
    """Drop one section's record. No-op if it doesn't exist."""
    rec = state.files.get(source_file)
    if rec is None:
        return
    rec.sections.pop(section_id, None)


def remove_file(state: State, source_file: str) -> None:
    """Drop the entire file record (case B: file removed from docs/<type>/en/)."""
    state.files.pop(source_file, None)
