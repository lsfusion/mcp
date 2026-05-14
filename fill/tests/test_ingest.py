"""Tests for fill.ingest — orchestration with a FakeVectorStoreClient.

Each test writes synthetic markdown into tmp_path, invokes `ingest_files`,
and asserts both the VS-side effects (via FakeVectorStoreClient introspection)
and the state mutations.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fill.ingest import ingest_files
from fill.openai_client import FakeVectorStoreClient
from fill.state import State
from fill.versions import pipeline_versions


def _write_md(path: Path, title: str, body: str) -> None:
    path.write_text(f"---\ntitle: {title}\n---\n\n{body}\n", encoding="utf-8")


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    d = tmp_path / "docs"
    d.mkdir()
    return d


def _src(p: Path, root: Path) -> str:
    return str(p.relative_to(root))


def _frozen_clock(stamp: str = "2026-05-14T12:00:00Z"):
    return lambda: stamp


# ─── case C: new files ──────────────────────────────────────────────────────


def test_new_file_uploads_all_sections(docs):
    p = docs / "AGGR.md"
    _write_md(p, "AGGR operator", "## Syntax\n\nfoo\n\n## Examples\n\nbar\n")
    state = State()
    client = FakeVectorStoreClient()

    stats = ingest_files(
        state,
        client,
        files_to_process=[p],
        files_removed=[],
        docs_root=docs,
        source_type_for=lambda _: "language",
        slug_for=lambda _: "AGGR",
        now=_frozen_clock(),
    )

    assert stats.files_processed == 1
    assert stats.files_fast_path_skipped == 0
    # Exact count: ## Syntax + ## Examples → 2 sections (the H1 has no
    # preamble paragraph, so the chunker doesn't emit a separate section for it).
    assert stats.sections_uploaded == 2
    assert stats.sections_uploaded == len(state.files[_src(p, docs)].sections)
    assert stats.sections_deleted == 0
    assert stats.errors == []

    rec = state.files[_src(p, docs)]
    assert rec.file_hash and rec.file_hash.startswith("sha256:")
    assert rec.indexed_with == pipeline_versions()
    assert rec.indexed_sourceType == "language"
    assert rec.indexed_at == "2026-05-14T12:00:00Z"
    assert rec.stale is False
    assert len(rec.sections) == stats.sections_uploaded
    # Every state section_id must match a stored fake file.
    for sid, srec in rec.sections.items():
        attrs = client.stored_attributes(srec.file_id)
        assert attrs["section_id"] == sid
        assert attrs["slug"] == "AGGR"
        assert attrs["sourceType"] == "language"
        assert attrs["section_payload_hash"] == srec.section_payload_hash


# ─── fast-path skip ────────────────────────────────────────────────────────


def test_fast_path_skips_unchanged_file(docs):
    p = docs / "AGGR.md"
    _write_md(p, "AGGR operator", "## Syntax\n\nfoo\n")
    state = State()
    client = FakeVectorStoreClient()

    # First ingest establishes baseline.
    ingest_files(
        state, client,
        files_to_process=[p], files_removed=[],
        docs_root=docs,
        source_type_for=lambda _: "language",
        slug_for=lambda _: "AGGR",
        now=_frozen_clock(),
    )
    uploads_before = len(client.upload_calls)

    # Re-run with same content, same versions, same sourceType — must skip.
    stats = ingest_files(
        state, client,
        files_to_process=[p], files_removed=[],
        docs_root=docs,
        source_type_for=lambda _: "language",
        slug_for=lambda _: "AGGR",
        now=_frozen_clock("2026-06-01T00:00:00Z"),
    )
    assert stats.files_fast_path_skipped == 1
    assert stats.files_processed == 0
    assert stats.sections_uploaded == 0
    assert stats.sections_deleted == 0
    assert len(client.upload_calls) == uploads_before  # no new uploads


def test_fast_path_busted_by_sourcetype_change(docs):
    p = docs / "AGGR.md"
    _write_md(p, "AGGR operator", "## Syntax\n\nfoo\n")
    state = State()
    client = FakeVectorStoreClient()
    common = dict(
        files_to_process=[p], files_removed=[],
        docs_root=docs,
        slug_for=lambda _: "AGGR",
        now=_frozen_clock(),
    )

    ingest_files(state, client, source_type_for=lambda _: "language", **common)
    initial_sections = dict(state.files[_src(p, docs)].sections)
    initial_file_ids = {srec.file_id for srec in initial_sections.values()}

    # Different sourceType → section_payload_hash differs (hash includes sourceType)
    # → every section must be re-uploaded AND the previous file_ids deleted.
    stats = ingest_files(state, client, source_type_for=lambda _: "paradigm", **common)
    assert stats.files_fast_path_skipped == 0
    assert stats.files_processed == 1
    assert stats.sections_uploaded == len(initial_sections)
    assert stats.sections_deleted == len(initial_sections)
    new_sections = state.files[_src(p, docs)].sections
    assert all(new_sections[sid].file_id != initial_sections[sid].file_id for sid in initial_sections)
    assert state.files[_src(p, docs)].indexed_sourceType == "paradigm"

    # VS-side: old file_ids gone, new uploads carry the new sourceType.
    live_ids = {vs.file_id for vs in client.list_sections()}
    assert initial_file_ids.isdisjoint(live_ids)
    new_uploads = client.upload_calls[len(initial_sections):]
    assert new_uploads and all(call["attributes"]["sourceType"] == "paradigm" for call in new_uploads)


# ─── case A: file content changed ──────────────────────────────────────────


def test_changed_file_reuploads_only_changed_sections(docs):
    p = docs / "AGGR.md"
    _write_md(p, "AGGR operator", "## Syntax\n\nfoo\n\n## Examples\n\nbar\n")
    state = State()
    client = FakeVectorStoreClient()
    common = dict(
        files_to_process=[p], files_removed=[],
        docs_root=docs,
        source_type_for=lambda _: "language",
        slug_for=lambda _: "AGGR",
        now=_frozen_clock(),
    )

    ingest_files(state, client, **common)
    initial_recs = {sid: srec.file_id for sid, srec in state.files[_src(p, docs)].sections.items()}

    # Modify only the "Examples" section — "Syntax" hash must stay.
    _write_md(p, "AGGR operator", "## Syntax\n\nfoo\n\n## Examples\n\nbar baz different\n")
    stats = ingest_files(state, client, **common)

    assert stats.files_processed == 1
    # At least one upload (the Examples section) and one delete (old Examples).
    assert stats.sections_uploaded >= 1
    assert stats.sections_deleted >= 1
    assert stats.errors == []

    final_recs = {sid: srec.file_id for sid, srec in state.files[_src(p, docs)].sections.items()}
    # Syntax section's file_id is preserved (unchanged hash → no reupload).
    syntax_sid = next(sid for sid in initial_recs if "syntax" in sid.lower())
    assert final_recs[syntax_sid] == initial_recs[syntax_sid]
    # Examples section got a new file_id.
    examples_sid = next(sid for sid in initial_recs if "examples" in sid.lower())
    assert final_recs[examples_sid] != initial_recs[examples_sid]

    # VS should not contain the old Examples file_id anymore.
    live_ids = {vs.file_id for vs in client.list_sections()}
    assert initial_recs[examples_sid] not in live_ids
    assert final_recs[examples_sid] in live_ids


def test_changed_file_drops_disappeared_sections(docs):
    p = docs / "AGGR.md"
    _write_md(p, "AGGR operator", "## Syntax\n\nfoo\n\n## Examples\n\nbar\n")
    state = State()
    client = FakeVectorStoreClient()
    common = dict(
        files_to_process=[p], files_removed=[],
        docs_root=docs,
        source_type_for=lambda _: "language",
        slug_for=lambda _: "AGGR",
        now=_frozen_clock(),
    )

    ingest_files(state, client, **common)
    initial_section_count = len(state.files[_src(p, docs)].sections)
    assert initial_section_count >= 2

    # Drop "Examples" entirely.
    _write_md(p, "AGGR operator", "## Syntax\n\nfoo\n")
    stats = ingest_files(state, client, **common)

    final_section_count = len(state.files[_src(p, docs)].sections)
    assert final_section_count < initial_section_count
    assert stats.sections_deleted >= 1
    assert all(
        "examples" not in sid.lower()
        for sid in state.files[_src(p, docs)].sections
    )


# ─── case B: file removed ───────────────────────────────────────────────────


def test_removed_file_deletes_all_sections_and_drops_record(docs):
    p = docs / "AGGR.md"
    _write_md(p, "AGGR operator", "## Syntax\n\nfoo\n\n## Examples\n\nbar\n")
    state = State()
    client = FakeVectorStoreClient()
    common = dict(
        files_to_process=[p], files_removed=[],
        docs_root=docs,
        source_type_for=lambda _: "language",
        slug_for=lambda _: "AGGR",
        now=_frozen_clock(),
    )

    ingest_files(state, client, **common)
    source_file = _src(p, docs)
    file_ids = [srec.file_id for srec in state.files[source_file].sections.values()]
    assert file_ids and all(fid in {vs.file_id for vs in client.list_sections()} for fid in file_ids)

    # File deleted from docs/.
    p.unlink()
    stats = ingest_files(
        state, client,
        files_to_process=[], files_removed=[source_file],
        docs_root=docs,
        source_type_for=lambda _: "language",
        slug_for=lambda _: "AGGR",
        now=_frozen_clock(),
    )

    assert stats.files_removed == 1
    assert stats.sections_deleted == len(file_ids)
    assert source_file not in state.files
    live_ids = {vs.file_id for vs in client.list_sections()}
    for fid in file_ids:
        assert fid not in live_ids


def test_removed_file_unknown_to_state_is_noop(docs):
    state = State()
    client = FakeVectorStoreClient()
    stats = ingest_files(
        state, client,
        files_to_process=[], files_removed=["never-tracked.md"],
        docs_root=docs,
        source_type_for=lambda _: "language",
        slug_for=lambda _: "x",
        now=_frozen_clock(),
    )
    assert stats.files_removed == 0
    assert stats.sections_deleted == 0
    assert stats.errors == []


# ─── failure handling ──────────────────────────────────────────────────────


def test_upload_failure_marks_file_stale(docs):
    p = docs / "AGGR.md"
    _write_md(p, "AGGR operator", "## Syntax\n\nfoo\n")
    state = State()
    client = FakeVectorStoreClient()
    # Refuse uploads for any section_id matching the H1.
    # We don't know the exact section_id yet — fail upload for everything matching the slug.
    # Simpler: fail the FIRST upload only by patching after we know structure.

    # First do a successful ingest to learn the section_ids.
    probe_client = FakeVectorStoreClient()
    ingest_files(
        State(), probe_client,
        files_to_process=[p], files_removed=[],
        docs_root=docs,
        source_type_for=lambda _: "language",
        slug_for=lambda _: "AGGR",
        now=_frozen_clock(),
    )
    sid_to_fail = probe_client.upload_calls[0]["attributes"]["section_id"]

    client.fail_upload_for_section_id.add(sid_to_fail)
    stats = ingest_files(
        state, client,
        files_to_process=[p], files_removed=[],
        docs_root=docs,
        source_type_for=lambda _: "language",
        slug_for=lambda _: "AGGR",
        now=_frozen_clock(),
    )
    assert any(sid_to_fail in err for err in stats.errors)
    assert state.files[_src(p, docs)].stale is True
    # Other sections should still have been uploaded.
    assert stats.sections_uploaded == len(probe_client.upload_calls) - 1


def test_old_delete_failure_marks_file_stale_but_keeps_new_upload(docs):
    p = docs / "AGGR.md"
    _write_md(p, "AGGR operator", "## Syntax\n\nfoo\n")
    state = State()
    client = FakeVectorStoreClient()
    common = dict(
        files_to_process=[p], files_removed=[],
        docs_root=docs,
        source_type_for=lambda _: "language",
        slug_for=lambda _: "AGGR",
        now=_frozen_clock(),
    )
    ingest_files(state, client, **common)
    original_file_ids = [srec.file_id for srec in state.files[_src(p, docs)].sections.values()]
    client.fail_delete_for_file_id.update(original_file_ids)

    _write_md(p, "AGGR operator", "## Syntax\n\nfoo modified\n")
    stats = ingest_files(state, client, **common)

    assert state.files[_src(p, docs)].stale is True
    assert any("delete-old" in err for err in stats.errors)
    # State must point at the NEW file_id even though old was orphaned.
    for srec in state.files[_src(p, docs)].sections.values():
        assert srec.file_id not in original_file_ids


def test_chunker_failure_marks_file_stale(docs, monkeypatch):
    p = docs / "AGGR.md"
    _write_md(p, "AGGR operator", "## Syntax\n\nfoo\n")
    state = State()
    client = FakeVectorStoreClient()

    import fill.ingest as ing
    def boom(*a, **kw):
        raise ValueError("synthetic chunker failure")
    monkeypatch.setattr(ing, "chunk_md", boom)

    stats = ingest_files(
        state, client,
        files_to_process=[p], files_removed=[],
        docs_root=docs,
        source_type_for=lambda _: "language",
        slug_for=lambda _: "AGGR",
        now=_frozen_clock(),
    )
    assert state.files[_src(p, docs)].stale is True
    assert any("synthetic chunker failure" in err for err in stats.errors)
    assert stats.files_processed == 0
    assert stats.sections_uploaded == 0


def test_read_failure_marks_file_stale(docs):
    """Unreadable file (path that exists per caller but read fails) →
    stale flag, no chunking, no uploads."""
    p = docs / "missing.md"
    # Don't create the file. ingest will try to read it.
    state = State()
    client = FakeVectorStoreClient()

    stats = ingest_files(
        state, client,
        files_to_process=[p], files_removed=[],
        docs_root=docs,
        source_type_for=lambda _: "language",
        slug_for=lambda _: "missing",
        now=_frozen_clock(),
    )
    assert state.files[_src(p, docs)].stale is True
    assert any("read" in err for err in stats.errors)
    assert stats.files_processed == 0


# ─── attribute payload ─────────────────────────────────────────────────────


def test_upload_attributes_include_all_required_keys(docs):
    p = docs / "AGGR.md"
    _write_md(p, "AGGR operator", "## Syntax\n\nfoo\n")
    state = State()
    client = FakeVectorStoreClient()

    ingest_files(
        state, client,
        files_to_process=[p], files_removed=[],
        docs_root=docs,
        source_type_for=lambda _: "language",
        slug_for=lambda _: "AGGR",
        now=_frozen_clock(),
    )
    required = {
        "section_id", "slug", "sourceType", "section_name",
        "heading_path", "source_url", "section_payload_hash",
        "source_file", "file_hash",
    }
    for call in client.upload_calls:
        assert required.issubset(call["attributes"].keys())
        # indexed_at intentionally NOT in attributes — see _section_attributes docs.
        assert "indexed_at" not in call["attributes"]


# ─── multi-file run ────────────────────────────────────────────────────────


def test_multi_file_independent_processing(docs):
    p1 = docs / "AGGR.md"
    p2 = docs / "OTHER.md"
    _write_md(p1, "AGGR", "## A\n\naaa\n")
    _write_md(p2, "OTHER", "## B\n\nbbb\n")

    state = State()
    client = FakeVectorStoreClient()

    stats = ingest_files(
        state, client,
        files_to_process=[p1, p2], files_removed=[],
        docs_root=docs,
        source_type_for=lambda path: "paradigm" if "OTHER" in path.name else "language",
        slug_for=lambda path: path.stem,
        now=_frozen_clock(),
    )

    assert stats.files_processed == 2
    assert _src(p1, docs) in state.files
    assert _src(p2, docs) in state.files
    assert state.files[_src(p1, docs)].indexed_sourceType == "language"
    assert state.files[_src(p2, docs)].indexed_sourceType == "paradigm"


# ─── fake client basics ─────────────────────────────────────────────────────


def test_fake_client_assigns_unique_ids():
    c = FakeVectorStoreClient()
    a = c.upload_section(content="x", filename="x.md", attributes={"section_id": "a"})
    b = c.upload_section(content="y", filename="y.md", attributes={"section_id": "b"})
    assert a != b
    assert c.file_count == 2


def test_fake_client_delete_unknown_raises():
    c = FakeVectorStoreClient()
    with pytest.raises(KeyError):
        c.delete_section("fake-file-99")


# ─── contract coverage (added in review round 1) ────────────────────────────


def test_stale_recovery_then_fast_path(docs):
    """The main lifecycle: upload fails → stale=True → next run rechunks +
    succeeds → stale=False → third run with no changes fast-paths."""
    p = docs / "AGGR.md"
    _write_md(p, "AGGR operator", "## Syntax\n\nfoo\n")
    state = State()
    client = FakeVectorStoreClient()
    common = dict(
        files_to_process=[p], files_removed=[],
        docs_root=docs,
        source_type_for=lambda _: "language",
        slug_for=lambda _: "AGGR",
        now=_frozen_clock(),
    )

    # Probe to learn section_ids and pick one to fail on round 1.
    probe = FakeVectorStoreClient()
    ingest_files(State(), probe, **common)
    fail_sid = probe.upload_calls[0]["attributes"]["section_id"]

    # Round 1: upload fails → stale.
    client.fail_upload_for_section_id.add(fail_sid)
    s1 = ingest_files(state, client, **common)
    assert state.files[_src(p, docs)].stale is True
    assert s1.files_fast_path_skipped == 0

    # Round 2: fault removed → reprocess, stale clears (no fast-path because of stale).
    client.fail_upload_for_section_id.clear()
    s2 = ingest_files(state, client, **common)
    assert state.files[_src(p, docs)].stale is False
    assert s2.files_fast_path_skipped == 0
    assert s2.files_processed == 1

    # Round 3: no changes → fast-path skip.
    s3 = ingest_files(state, client, **common)
    assert s3.files_fast_path_skipped == 1
    assert s3.files_processed == 0
    assert s3.sections_uploaded == 0


def test_one_file_failure_does_not_block_others(docs):
    """If file1's upload fails, file2 must still be fully processed."""
    p1 = docs / "AGGR.md"
    p2 = docs / "OTHER.md"
    _write_md(p1, "AGGR", "## A\n\naaa\n")
    _write_md(p2, "OTHER", "## B\n\nbbb\n")

    state = State()
    client = FakeVectorStoreClient()
    common = dict(
        files_to_process=[p1, p2], files_removed=[],
        docs_root=docs,
        source_type_for=lambda _: "language",
        slug_for=lambda path: path.stem,
        now=_frozen_clock(),
    )

    # Probe file1's section_ids and fail one of them.
    probe = FakeVectorStoreClient()
    ingest_files(State(), probe, **common)
    p1_section_ids = {
        str(call["attributes"]["section_id"])
        for call in probe.upload_calls
        if str(call["attributes"]["slug"]) == "AGGR"
    }
    client.fail_upload_for_section_id.update(p1_section_ids)

    stats = ingest_files(state, client, **common)
    # file1 partially failed → stale.
    assert state.files[_src(p1, docs)].stale is True
    # file2 fully succeeded → not stale, sections recorded.
    assert state.files[_src(p2, docs)].stale is False
    assert len(state.files[_src(p2, docs)].sections) >= 1
    # File2's processed-count still bumped.
    assert stats.files_processed == 2
    assert any(_src(p1, docs) in err for err in stats.errors)


def test_removed_file_partial_delete_failure_keeps_record(docs):
    """If one section's delete fails during Case B, the FileRecord stays in
    state (marked stale) so the next run can retry."""
    p = docs / "AGGR.md"
    _write_md(p, "AGGR operator", "## Syntax\n\nfoo\n\n## Examples\n\nbar\n")
    state = State()
    client = FakeVectorStoreClient()
    common = dict(
        files_to_process=[p], files_removed=[],
        docs_root=docs,
        source_type_for=lambda _: "language",
        slug_for=lambda _: "AGGR",
        now=_frozen_clock(),
    )
    ingest_files(state, client, **common)
    source_file = _src(p, docs)
    file_ids = [srec.file_id for srec in state.files[source_file].sections.values()]
    assert len(file_ids) >= 2

    # Fail delete on exactly one of the file_ids.
    client.fail_delete_for_file_id.add(file_ids[0])

    p.unlink()
    stats = ingest_files(
        state, client,
        files_to_process=[], files_removed=[source_file],
        docs_root=docs,
        source_type_for=lambda _: "language",
        slug_for=lambda _: "AGGR",
        now=_frozen_clock(),
    )

    # FileRecord retained, marked stale; some sections deleted, the failed one kept.
    assert source_file in state.files
    assert state.files[source_file].stale is True
    assert stats.files_removed == 0
    assert any("delete (removed file)" in err for err in stats.errors)
    remaining_file_ids = {srec.file_id for srec in state.files[source_file].sections.values()}
    assert file_ids[0] in remaining_file_ids
    assert file_ids[1] not in remaining_file_ids


def test_setup_error_does_not_abort_run(docs):
    """If `source_type_for` raises for file1, the run continues to file2."""
    p1 = docs / "AGGR.md"
    p2 = docs / "OTHER.md"
    _write_md(p1, "AGGR", "## A\n\naaa\n")
    _write_md(p2, "OTHER", "## B\n\nbbb\n")
    state = State()
    client = FakeVectorStoreClient()

    def source_type_for(path: Path):
        if path.name == "AGGR.md":
            raise KeyError("manifest miss")
        return "language"

    stats = ingest_files(
        state, client,
        files_to_process=[p1, p2], files_removed=[],
        docs_root=docs,
        source_type_for=source_type_for,
        slug_for=lambda path: path.stem,
        now=_frozen_clock(),
    )

    assert any("setup" in err and "AGGR.md" in err for err in stats.errors)
    assert state.files[_src(p1, docs)].stale is True
    # file2 still processed.
    assert _src(p2, docs) in state.files
    assert state.files[_src(p2, docs)].stale is False


def test_parallel_upload_assigns_all_sections(docs):
    """Many small files, max_workers=8 — every section must end up with a
    unique file_id in state, and the count must match. No collisions, no
    drops. Verifies the FakeVectorStoreClient lock + the as_completed
    consumer pattern."""
    # Build a fixture with enough sections to actually exercise the pool.
    for i in range(20):
        _write_md(
            docs / f"DOC{i:02d}.md",
            f"DOC{i:02d}",
            "\n".join(f"## S{j}\n\nbody-{i}-{j}\n" for j in range(4)),
        )
    state = State()
    client = FakeVectorStoreClient()
    paths = sorted(docs.glob("*.md"))

    stats = ingest_files(
        state, client,
        files_to_process=paths, files_removed=[],
        docs_root=docs,
        source_type_for=lambda _: "language",
        slug_for=lambda p: p.stem,
        now=_frozen_clock(),
        max_workers=8,
    )

    assert stats.errors == []
    assert stats.files_processed == 20
    # Every section uploaded once, each got a unique file_id, and every
    # file_id recorded in state exists in the fake VS.
    expected_sections = 20 * 4  # one Section per `## S{j}` heading
    assert stats.sections_uploaded == expected_sections
    all_file_ids = {
        srec.file_id
        for rec in state.files.values()
        for srec in rec.sections.values()
    }
    assert len(all_file_ids) == expected_sections
    live = {vs.file_id for vs in client.list_sections()}
    assert all_file_ids == live


def test_max_workers_zero_raises(docs):
    state = State()
    client = FakeVectorStoreClient()
    with pytest.raises(ValueError, match="max_workers"):
        ingest_files(
            state, client,
            files_to_process=[], files_removed=[],
            docs_root=docs,
            source_type_for=lambda _: "language",
            slug_for=lambda _: "x",
            now=_frozen_clock(),
            max_workers=0,
        )


def test_max_workers_one_is_serial_and_still_correct(docs):
    """Single-worker mode disables parallelism but must still produce the
    same state shape. Useful for debugging or rate-limited environments."""
    p = docs / "AGGR.md"
    _write_md(p, "AGGR", "## A\n\naaa\n\n## B\n\nbbb\n")
    state = State()
    client = FakeVectorStoreClient()
    stats = ingest_files(
        state, client,
        files_to_process=[p], files_removed=[],
        docs_root=docs,
        source_type_for=lambda _: "language",
        slug_for=lambda _: "AGGR",
        now=_frozen_clock(),
        max_workers=1,
    )
    assert stats.errors == []
    assert stats.sections_uploaded == 2
    assert len(state.files["AGGR.md"].sections) == 2


def test_filename_drops_redundant_slug_prefix(docs):
    """section_id already starts with slug, so the filename uses just
    section_id (with `::` → `__`). Catches a regression where the slug
    was prepended a second time, producing `AGGR--AGGR__syntax.md`."""
    p = docs / "AGGR.md"
    _write_md(p, "AGGR", "## Syntax\n\nfoo\n")
    state = State()
    client = FakeVectorStoreClient()
    ingest_files(
        state, client,
        files_to_process=[p], files_removed=[],
        docs_root=docs,
        source_type_for=lambda _: "language",
        slug_for=lambda _: "AGGR",
        now=_frozen_clock(),
    )
    for call in client.upload_calls:
        filename = call["filename"]
        # Section ids look like "AGGR::syntax" → filename "AGGR__syntax.md".
        assert "--" not in filename, f"unexpected `--` in {filename}"
        assert filename.count("AGGR") == 1, f"slug duplicated: {filename}"


def test_attributes_idempotent_across_unchanged_reuploads(docs):
    """Same content → same attributes byte-for-byte (no `indexed_at` field)."""
    p = docs / "AGGR.md"
    _write_md(p, "AGGR operator", "## Syntax\n\nfoo\n")
    common = dict(
        files_to_process=[p], files_removed=[],
        docs_root=docs,
        source_type_for=lambda _: "language",
        slug_for=lambda _: "AGGR",
    )
    s1 = State()
    c1 = FakeVectorStoreClient()
    ingest_files(s1, c1, now=_frozen_clock("2026-05-14T12:00:00Z"), **common)
    s2 = State()
    c2 = FakeVectorStoreClient()
    ingest_files(s2, c2, now=_frozen_clock("2099-01-01T00:00:00Z"), **common)
    # Different clocks but identical attributes (indexed_at intentionally absent).
    a1 = [call["attributes"] for call in c1.upload_calls]
    a2 = [call["attributes"] for call in c2.upload_calls]
    assert a1 == a2
