"""Tests for fill.state — typed load/save + atomic write + CRUD."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from fill.state import (
    FileRecord,
    MAX_STATE_BYTES,
    SectionRecord,
    State,
    load,
    mark_stale,
    remove_file,
    remove_section,
    save,
    set_reconcile_sentinel,
    upsert_section,
)


# ─── load ────────────────────────────────────────────────────────────────────


def test_load_missing_file_returns_empty_state(tmp_path):
    s = load(tmp_path / "no-such.json")
    assert s.vector_store_id is None
    assert s.last_indexed_docs_commit is None
    assert s.pipeline_versions is None
    assert s.files == {}


def test_load_round_trip(tmp_path):
    src = State(
        vector_store_id="vs_xxx",
        last_indexed_docs_commit="abc123",
        pipeline_versions={
            "chunker_version": "v1",
            "glossary_version": "v0",
            "prefix_version": "v1",
            "source_url_version": "v1",
        },
        files={
            "docs/en/AGGR_operator.md": FileRecord(
                file_hash="sha256:hash1",
                indexed_with={
                    "chunker_version": "v1",
                    "glossary_version": "v0",
                    "prefix_version": "v1",
                    "source_url_version": "v1",
                },
                indexed_sourceType="language",
                indexed_at="2026-05-14T12:34:56Z",
                stale=False,
                sections={
                    "AGGR_operator::syntax": SectionRecord(
                        file_id="file-abc",
                        section_payload_hash="sha256:secs1",
                        heading_path="AGGR_operator > Syntax",
                        source_file="docs/en/AGGR_operator.md",
                    ),
                    "AGGR_operator::examples": SectionRecord(
                        file_id="file-def",
                        section_payload_hash="sha256:secs2",
                        heading_path="AGGR_operator > Examples",
                        source_file="docs/en/AGGR_operator.md",
                    ),
                },
            ),
        },
    )
    p = tmp_path / "state.json"
    save(p, src)
    loaded = load(p)
    assert loaded == src


def test_load_invalid_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("not json {{")
    with pytest.raises(ValueError, match="invalid JSON"):
        load(p)


def test_load_root_not_object(tmp_path):
    p = tmp_path / "arr.json"
    p.write_text("[]")
    with pytest.raises(ValueError, match="must be a JSON object"):
        load(p)


def test_load_rejects_non_regular_file(tmp_path):
    # Simulate a non-regular file via /dev/null (if available) or a fifo.
    fifo = tmp_path / "fifo"
    try:
        os.mkfifo(fifo)
    except (OSError, AttributeError):
        pytest.skip("mkfifo not available")
    with pytest.raises(ValueError, match="not a regular file"):
        load(fifo)


def test_load_rejects_oversize(tmp_path):
    p = tmp_path / "huge.json"
    p.write_text("{" + " " * 100 + "}")  # 102 bytes
    import fill.state as st
    orig = st.MAX_STATE_BYTES
    st.MAX_STATE_BYTES = 50  # smaller than the actual file
    try:
        with pytest.raises(ValueError, match="exceeds cap"):
            load(p)
    finally:
        st.MAX_STATE_BYTES = orig


def test_load_rejects_missing_section_fields(tmp_path):
    p = tmp_path / "bad-section.json"
    p.write_text(
        json.dumps(
            {
                "files": {
                    "docs/en/X.md": {
                        "sections": {
                            "X::foo": {
                                "file_id": "file-1",
                                # missing section_payload_hash
                                "heading_path": "X > Foo",
                                "source_file": "docs/en/X.md",
                            }
                        }
                    }
                }
            }
        )
    )
    with pytest.raises(ValueError, match="section_payload_hash"):
        load(p)


def test_load_rejects_non_string_vector_store_id(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"vector_store_id": 42}))
    with pytest.raises(ValueError, match="vector_store_id"):
        load(p)


def test_load_accepts_null_sentinels(tmp_path):
    # pipeline_versions=null + last_indexed_docs_commit=null = reconcile sentinel
    p = tmp_path / "sentinel.json"
    p.write_text(json.dumps({
        "vector_store_id": "vs_xxx",
        "last_indexed_docs_commit": None,
        "pipeline_versions": None,
        "files": {},
    }))
    s = load(p)
    assert s.last_indexed_docs_commit is None
    assert s.pipeline_versions is None


# ─── save / atomic ───────────────────────────────────────────────────────────


def test_save_creates_file_atomically(tmp_path):
    p = tmp_path / "state.json"
    save(p, State(vector_store_id="vs_test"))
    assert p.exists()
    assert not (tmp_path / "state.json.tmp").exists()  # tmp removed by os.replace


def test_save_overwrites_existing(tmp_path):
    p = tmp_path / "state.json"
    save(p, State(vector_store_id="vs_a"))
    save(p, State(vector_store_id="vs_b"))
    assert load(p).vector_store_id == "vs_b"


def test_save_creates_parent_dirs(tmp_path):
    p = tmp_path / "deeply" / "nested" / "state.json"
    save(p, State())
    assert p.exists()


def test_save_serializes_human_readable_json(tmp_path):
    p = tmp_path / "state.json"
    save(p, State(vector_store_id="vs_xxx"))
    text = p.read_text()
    assert "\n" in text  # indent=2
    assert "vs_xxx" in text


# ─── sentinels / drift / stale ───────────────────────────────────────────────


def test_needs_forced_full_scan_on_no_commit():
    assert State(last_indexed_docs_commit=None, pipeline_versions={"x": "1"}).needs_forced_full_scan({"x": "1"})


def test_needs_forced_full_scan_on_no_versions():
    assert State(last_indexed_docs_commit="abc", pipeline_versions=None).needs_forced_full_scan({"x": "1"})


def test_needs_forced_full_scan_on_version_drift():
    s = State(last_indexed_docs_commit="abc", pipeline_versions={"chunker_version": "v1"})
    assert s.needs_forced_full_scan({"chunker_version": "v2"})


def test_does_not_need_full_scan_on_match():
    s = State(last_indexed_docs_commit="abc", pipeline_versions={"chunker_version": "v1"})
    assert not s.needs_forced_full_scan({"chunker_version": "v1"})


def test_stale_files():
    s = State(files={
        "a.md": FileRecord(stale=True),
        "b.md": FileRecord(stale=False),
        "c.md": FileRecord(stale=True),
    })
    assert s.stale_files() == {"a.md", "c.md"}


def test_set_reconcile_sentinel():
    s = State(last_indexed_docs_commit="abc", pipeline_versions={"x": "1"})
    set_reconcile_sentinel(s)
    assert s.last_indexed_docs_commit is None
    assert s.pipeline_versions is None


def test_mark_stale_existing_file():
    s = State(files={"a.md": FileRecord(file_hash="h", stale=False)})
    mark_stale(s, "a.md")
    assert s.files["a.md"].stale is True
    assert s.files["a.md"].file_hash == "h"  # preserves other fields


def test_mark_stale_unknown_file_creates_stub():
    s = State()
    mark_stale(s, "new.md")
    assert s.files["new.md"].stale is True
    assert s.files["new.md"].file_hash is None


# ─── fast-path skip predicate ───────────────────────────────────────────────


@pytest.fixture
def good_state():
    return State(
        files={
            "a.md": FileRecord(
                file_hash="sha256:h1",
                indexed_with={"chunker_version": "v1"},
                indexed_sourceType="language",
                stale=False,
            ),
        },
    )


def test_fast_path_skip_happy(good_state):
    assert good_state.can_skip_file_fast_path(
        "a.md", "sha256:h1", {"chunker_version": "v1"}, "language"
    )


def test_fast_path_skip_file_unknown(good_state):
    assert not good_state.can_skip_file_fast_path(
        "missing.md", "sha256:h1", {"chunker_version": "v1"}, "language"
    )


def test_fast_path_skip_stale(good_state):
    good_state.files["a.md"].stale = True
    assert not good_state.can_skip_file_fast_path(
        "a.md", "sha256:h1", {"chunker_version": "v1"}, "language"
    )


def test_fast_path_skip_file_hash_mismatch(good_state):
    assert not good_state.can_skip_file_fast_path(
        "a.md", "sha256:DIFFERENT", {"chunker_version": "v1"}, "language"
    )


def test_fast_path_skip_versions_mismatch(good_state):
    assert not good_state.can_skip_file_fast_path(
        "a.md", "sha256:h1", {"chunker_version": "v2"}, "language"
    )


def test_fast_path_skip_indexed_with_null(good_state):
    # `indexed_with=None` from reconcile means "unknown" → never fast-path.
    good_state.files["a.md"].indexed_with = None
    assert not good_state.can_skip_file_fast_path(
        "a.md", "sha256:h1", {"chunker_version": "v1"}, "language"
    )


def test_fast_path_skip_sourcetype_mismatch(good_state):
    assert not good_state.can_skip_file_fast_path(
        "a.md", "sha256:h1", {"chunker_version": "v1"}, "paradigm"
    )


# ─── CRUD on sections ──────────────────────────────────────────────────────


def _section(file_id="file-x", section_id="A::foo"):
    return SectionRecord(
        file_id=file_id,
        section_payload_hash=f"sha256:{file_id}",
        heading_path="A > Foo",
        source_file="A.md",
    )


def test_upsert_section_creates_file_record_if_missing():
    s = State()
    upsert_section(s, "A.md", "A::foo", _section())
    assert "A.md" in s.files
    assert s.files["A.md"].sections["A::foo"].file_id == "file-x"


def test_upsert_section_replaces_existing():
    s = State(files={"A.md": FileRecord(sections={"A::foo": _section("file-old")})})
    upsert_section(s, "A.md", "A::foo", _section("file-new"))
    assert s.files["A.md"].sections["A::foo"].file_id == "file-new"


def test_remove_section_noop_on_unknown():
    s = State()
    remove_section(s, "A.md", "A::foo")  # should not raise


def test_remove_section_drops_only_target():
    s = State(
        files={
            "A.md": FileRecord(
                sections={
                    "A::foo": _section("file-1"),
                    "A::bar": _section("file-2"),
                }
            )
        }
    )
    remove_section(s, "A.md", "A::foo")
    assert "A::foo" not in s.files["A.md"].sections
    assert "A::bar" in s.files["A.md"].sections


def test_remove_file_drops_entire_record():
    s = State(files={"A.md": FileRecord(sections={"A::foo": _section()})})
    remove_file(s, "A.md")
    assert "A.md" not in s.files


# ─── schema fidelity round-trip ─────────────────────────────────────────────


def test_save_load_preserves_sentinel_nulls(tmp_path):
    p = tmp_path / "s.json"
    s = State()
    set_reconcile_sentinel(s)
    save(p, s)
    loaded = load(p)
    assert loaded.last_indexed_docs_commit is None
    assert loaded.pipeline_versions is None


def test_round_trip_preserves_stale_flag(tmp_path):
    p = tmp_path / "s.json"
    s = State(files={"a.md": FileRecord(stale=True, file_hash="h")})
    save(p, s)
    assert load(p).files["a.md"].stale is True


def test_round_trip_preserves_indexed_with_null(tmp_path):
    p = tmp_path / "s.json"
    s = State(files={"a.md": FileRecord(indexed_with=None, file_hash="h")})
    save(p, s)
    assert load(p).files["a.md"].indexed_with is None


# ─── strict schema validation (from review) ────────────────────────────────


def test_load_rejects_non_string_indexed_with_values(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({
        "files": {"a.md": {"indexed_with": {"chunker_version": 1}}}
    }))
    with pytest.raises(ValueError, match="indexed_with"):
        load(p)


def test_load_rejects_non_string_pipeline_versions_values(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"pipeline_versions": {"chunker_version": 1}}))
    with pytest.raises(ValueError, match="pipeline_versions"):
        load(p)


def test_load_rejects_non_bool_stale(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"files": {"a.md": {"stale": "yes"}}}))
    with pytest.raises(ValueError, match="stale must be bool"):
        load(p)


def test_load_rejects_null_files(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"files": None}))
    with pytest.raises(ValueError, match="state.files must be an object"):
        load(p)


def test_load_rejects_null_sections(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"files": {"a.md": {"sections": None}}}))
    with pytest.raises(ValueError, match="sections must be an object"):
        load(p)


def test_load_indexed_with_is_defensive_copy(tmp_path):
    """Loaded `indexed_with` must not share identity with the parsed JSON
    tree — otherwise downstream mutation could corrupt nothing visible, but
    catches a class of latent aliasing bugs in CRUD flows."""
    p = tmp_path / "s.json"
    save(p, State(files={"a.md": FileRecord(indexed_with={"x": "1"})}))
    a = load(p).files["a.md"].indexed_with
    b = load(p).files["a.md"].indexed_with
    assert a == b
    assert a is not b  # two independent loads → two independent dicts


def test_save_unlinks_tmp_on_replace_failure(tmp_path, monkeypatch):
    p = tmp_path / "state.json"
    save(p, State(vector_store_id="vs_initial"))  # establish prior good state

    # Force os.replace to fail after the .tmp is written.
    import fill.state as st
    def boom(*a, **kw):
        raise OSError("simulated rename failure")
    monkeypatch.setattr(st.os, "replace", boom)

    with pytest.raises(OSError, match="simulated"):
        save(p, State(vector_store_id="vs_doomed"))

    # Prior state preserved + no leftover .tmp.
    assert load(p).vector_store_id == "vs_initial"
    assert not (tmp_path / "state.json.tmp").exists()


def test_save_deterministic_bytes(tmp_path):
    """Same State → identical bytes (sort_keys=True)."""
    s = State(
        vector_store_id="vs_xxx",
        pipeline_versions={"b": "1", "a": "2"},
        files={"a.md": FileRecord(file_hash="h", indexed_with={"y": "2", "x": "1"})},
    )
    p1 = tmp_path / "s1.json"
    p2 = tmp_path / "s2.json"
    save(p1, s)
    save(p2, s)
    assert p1.read_bytes() == p2.read_bytes()
