"""Tests for fill.reconcile — rebuild state from a Vector Store listing."""

from __future__ import annotations

from fill.openai_client import FakeVectorStoreClient
from fill.reconcile import reconcile
from fill.state import FileRecord, SectionRecord, State


def _attrs(
    section_id: str,
    *,
    source_file: str = "AGGR_operator.md",
    payload_hash: str = "sha256:hash",
    heading_path: str = "AGGR > X",
    source_type: str | None = "language",
    extra: dict | None = None,
) -> dict:
    out: dict = {
        "section_id": section_id,
        "section_payload_hash": payload_hash,
        "heading_path": heading_path,
        "source_file": source_file,
    }
    if source_type is not None:
        out["sourceType"] = source_type
    if extra:
        out.update(extra)
    return out


# ─── happy paths ─────────────────────────────────────────────────────────────


def test_empty_vs_clears_state_and_sets_sentinels():
    state = State(
        vector_store_id="vs_keep_me",
        last_indexed_docs_commit="abc123",
        pipeline_versions={"x": "1"},
        files={"old.md": FileRecord(file_hash="h", sections={"old::s": SectionRecord(
            file_id="file-old", section_payload_hash="sha256:o",
            heading_path="o", source_file="old.md")})},
    )
    stats = reconcile(state, FakeVectorStoreClient())

    assert stats.files_in_vs == 0
    assert stats.source_files_rebuilt == 0
    assert state.files == {}  # old file dropped
    assert state.vector_store_id == "vs_keep_me"  # preserved
    assert state.last_indexed_docs_commit is None
    assert state.pipeline_versions is None
    assert state.needs_forced_full_scan({"x": "1"})  # sentinels engaged


def test_well_formed_vs_rebuilds_state():
    client = FakeVectorStoreClient()
    client.preload("file-1", _attrs(
        "AGGR::syntax", payload_hash="sha256:a", source_file="AGGR.md"))
    client.preload("file-2", _attrs(
        "AGGR::examples", payload_hash="sha256:b", source_file="AGGR.md"))
    client.preload("file-3", _attrs(
        "OTHER::body", payload_hash="sha256:c",
        source_file="OTHER.md", source_type="paradigm"))

    state = State(vector_store_id="vs_x")
    stats = reconcile(state, client)

    assert stats.files_in_vs == 3
    assert stats.sections_rebuilt == 3
    assert stats.source_files_rebuilt == 2
    assert stats.malformed_in_vs == []
    assert stats.duplicate_slots == []

    aggr = state.files["AGGR.md"]
    assert aggr.indexed_sourceType == "language"
    assert aggr.indexed_with is None
    assert aggr.file_hash is None
    assert aggr.indexed_at is None
    assert aggr.stale is True
    assert set(aggr.sections.keys()) == {"AGGR::syntax", "AGGR::examples"}
    assert aggr.sections["AGGR::syntax"].file_id == "file-1"
    assert aggr.sections["AGGR::syntax"].section_payload_hash == "sha256:a"

    other = state.files["OTHER.md"]
    assert other.indexed_sourceType == "paradigm"
    assert other.sections["OTHER::body"].file_id == "file-3"


# ─── malformed VS files ──────────────────────────────────────────────────────


def test_missing_required_attr_marks_malformed():
    client = FakeVectorStoreClient()
    client.preload("file-good", _attrs("AGGR::syntax"))
    # Missing section_payload_hash → malformed.
    bad = _attrs("AGGR::broken")
    del bad["section_payload_hash"]
    client.preload("file-bad", bad)

    state = State()
    stats = reconcile(state, client)

    assert stats.malformed_in_vs == ["file-bad"]
    assert stats.sections_rebuilt == 1
    assert "AGGR::broken" not in state.files["AGGR_operator.md"].sections


def test_non_string_attr_marks_malformed():
    client = FakeVectorStoreClient()
    attrs = _attrs("AGGR::x")
    attrs["section_id"] = 42  # type: ignore[assignment]
    client.preload("file-bad", attrs)

    state = State()
    stats = reconcile(state, client)

    assert stats.malformed_in_vs == ["file-bad"]
    assert state.files == {}


# ─── duplicate slots ─────────────────────────────────────────────────────────


def test_duplicate_section_id_keeps_smallest_file_id():
    client = FakeVectorStoreClient()
    client.preload("file-zz", _attrs(
        "AGGR::syntax", payload_hash="sha256:zz", source_file="AGGR.md"))
    client.preload("file-aa", _attrs(
        "AGGR::syntax", payload_hash="sha256:aa", source_file="AGGR.md"))
    client.preload("file-mm", _attrs(
        "AGGR::syntax", payload_hash="sha256:mm", source_file="AGGR.md"))

    state = State()
    stats = reconcile(state, client)

    winner = state.files["AGGR.md"].sections["AGGR::syntax"]
    assert winner.file_id == "file-aa"
    assert winner.section_payload_hash == "sha256:aa"

    assert len(stats.duplicate_slots) == 1
    src, sid, losers = stats.duplicate_slots[0]
    assert src == "AGGR.md"
    assert sid == "AGGR::syntax"
    # Deterministic order: candidates sorted ascending by file_id; losers
    # are the second+ entries.
    assert losers == ["file-mm", "file-zz"]
    # Read-only-on-VS contract: reconcile must never delete.
    assert client.delete_calls == []


# ─── sentinels / forced-full-scan ────────────────────────────────────────────


def test_reconcile_always_sets_sentinels():
    client = FakeVectorStoreClient()
    client.preload("file-1", _attrs("AGGR::s"))

    state = State(
        last_indexed_docs_commit="abc",
        pipeline_versions={"chunker_version": "v1"},
    )
    reconcile(state, client)
    assert state.last_indexed_docs_commit is None
    assert state.pipeline_versions is None
    # needs_forced_full_scan returns True for ANY current_versions.
    assert state.needs_forced_full_scan({"chunker_version": "v1"})
    assert state.needs_forced_full_scan({"chunker_version": "v999"})


def test_rebuilt_files_never_fast_path():
    client = FakeVectorStoreClient()
    client.preload("file-1", _attrs(
        "AGGR::syntax", source_file="AGGR.md"))
    state = State()
    reconcile(state, client)
    rec = state.files["AGGR.md"]
    # With indexed_with=None + stale=True, fast-path skip is impossible.
    assert not state.can_skip_file_fast_path(
        "AGGR.md", "sha256:anyhash", {"chunker_version": "v1"}, "language"
    )


# ─── source_type preservation ─────────────────────────────────────────────────


def test_missing_sourcetype_results_in_none():
    client = FakeVectorStoreClient()
    client.preload("file-1", _attrs(
        "AGGR::s", source_type=None))
    state = State()
    reconcile(state, client)
    assert state.files["AGGR_operator.md"].indexed_sourceType is None


def test_sourcetype_picked_deterministically_when_attrs_conflict():
    """If two VS files for the same source_file disagree on sourceType (a
    data bug), the winner is the lexicographically smallest file_id —
    same rule as duplicate-slot tiebreaking."""
    client = FakeVectorStoreClient()
    client.preload("file-zz", _attrs(
        "AGGR::a", source_file="AGGR.md", source_type="paradigm"))
    client.preload("file-aa", _attrs(
        "AGGR::b", source_file="AGGR.md", source_type="language"))
    state = State()
    reconcile(state, client)
    assert state.files["AGGR.md"].indexed_sourceType == "language"


# ─── state replacement semantics ─────────────────────────────────────────────


def test_old_state_files_not_in_vs_are_dropped():
    """Stale FileRecord for a file no longer in VS must be removed.
    The forced-full-scan sentinel ensures the next ingest revisits docs/."""
    state = State(
        files={
            "leftover.md": FileRecord(
                file_hash="sha256:h",
                stale=False,
                sections={"leftover::a": SectionRecord(
                    file_id="file-stale",
                    section_payload_hash="sha256:h",
                    heading_path="x",
                    source_file="leftover.md",
                )},
            ),
        },
    )
    client = FakeVectorStoreClient()
    client.preload("file-new", _attrs("X::y", source_file="X.md"))
    reconcile(state, client)
    assert "leftover.md" not in state.files
    assert "X.md" in state.files


def test_reconcile_preserves_vector_store_id():
    state = State(vector_store_id="vs_xyz")
    reconcile(state, FakeVectorStoreClient())
    assert state.vector_store_id == "vs_xyz"


# ─── empty-string and non-string attribute rejection ────────────────────────


def test_empty_string_source_file_is_malformed():
    client = FakeVectorStoreClient()
    client.preload("file-bad", _attrs("AGGR::s", source_file=""))
    state = State()
    stats = reconcile(state, client)
    assert stats.malformed_in_vs == ["file-bad"]
    assert state.files == {}


def test_empty_string_section_id_is_malformed():
    client = FakeVectorStoreClient()
    client.preload("file-bad", _attrs(""))
    state = State()
    stats = reconcile(state, client)
    assert stats.malformed_in_vs == ["file-bad"]
    assert state.files == {}


def test_non_string_sourcetype_silently_becomes_none():
    """A bad sourceType value must not crash reconcile; we leave None and
    let the next ingest re-stamp from the manifest."""
    client = FakeVectorStoreClient()
    attrs = _attrs("AGGR::s")
    attrs["sourceType"] = 42  # type: ignore[assignment]
    client.preload("file-1", attrs)
    state = State()
    reconcile(state, client)
    assert state.files["AGGR_operator.md"].indexed_sourceType is None


def test_only_malformed_files_still_sets_sentinels():
    """Zero usable files + N malformed: state.files stays empty, sentinels
    set, malformed list captures every bad file_id in listing order."""
    client = FakeVectorStoreClient()
    bad1 = _attrs("X::a")
    del bad1["section_payload_hash"]
    bad2 = _attrs("Y::b")
    del bad2["heading_path"]
    client.preload("file-bad-1", bad1)
    client.preload("file-bad-2", bad2)

    state = State(last_indexed_docs_commit="abc", pipeline_versions={"x": "1"})
    stats = reconcile(state, client)

    assert state.files == {}
    assert state.last_indexed_docs_commit is None
    assert state.pipeline_versions is None
    # Order is observable: malformed_in_vs preserves VS listing order.
    assert stats.malformed_in_vs == ["file-bad-1", "file-bad-2"]
    assert stats.sections_rebuilt == 0


def test_reconcile_never_calls_delete():
    """Read-only-on-VS contract — must hold across every test scenario."""
    client = FakeVectorStoreClient()
    client.preload("file-1", _attrs("AGGR::s"))
    bad = _attrs("Y::b")
    del bad["heading_path"]
    client.preload("file-bad", bad)
    state = State(
        files={"leftover.md": FileRecord(
            sections={"leftover::a": SectionRecord(
                file_id="file-leftover",
                section_payload_hash="sha256:o",
                heading_path="o",
                source_file="leftover.md",
            )},
        )},
    )
    reconcile(state, client)
    assert client.delete_calls == []


# ─── integration: reconcile then ingest restores cleanly ─────────────────────


def test_reconcile_then_ingest_round_trips(tmp_path):
    """After reconcile, when the caller passes the file to ingest, the
    fast-path must NOT kick in (stale=True + indexed_with=None gates it).
    This does not validate that the *caller* of `ragIngestDocs` would
    schedule the file from sentinels in Step 0 — that wiring lives outside
    fill.ingest and is tested separately when it lands."""
    from fill.ingest import ingest_files

    docs = tmp_path / "docs"
    docs.mkdir()
    p = docs / "AGGR.md"
    p.write_text("---\ntitle: AGGR operator\n---\n\n## Syntax\n\nfoo\n", encoding="utf-8")

    state = State()
    client = FakeVectorStoreClient()

    # Initial ingest.
    ingest_files(
        state, client,
        files_to_process=[p], files_removed=[],
        docs_root=docs,
        source_type_for=lambda _: "language",
        slug_for=lambda _: "AGGR",
        now=lambda: "2026-05-14T12:00:00Z",
    )

    # Simulate state loss: blow it away while VS is intact.
    state = State(vector_store_id="vs_x")
    reconcile_stats = reconcile(state, client)
    assert reconcile_stats.source_files_rebuilt == 1
    assert state.files["AGGR.md"].stale is True

    # Next ingest must NOT fast-path; the file should be reprocessed.
    stats = ingest_files(
        state, client,
        files_to_process=[p], files_removed=[],
        docs_root=docs,
        source_type_for=lambda _: "language",
        slug_for=lambda _: "AGGR",
        now=lambda: "2026-05-14T13:00:00Z",
    )
    assert stats.files_fast_path_skipped == 0
    assert stats.files_processed == 1
    # FileRecord is back to a normal shape: file_hash present, indexed_with
    # stamped, not stale. The root `pipeline_versions` is still None — only
    # the orchestrator stamps that at end-of-cycle (Step ∞).
    assert state.pipeline_versions is None
    rec = state.files["AGGR.md"]
    assert rec.file_hash is not None
    assert rec.indexed_with is not None
    assert rec.stale is False
