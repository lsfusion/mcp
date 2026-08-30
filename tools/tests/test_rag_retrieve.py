"""Tests for tools/rag_retrieve.py — branch coverage + type validation.

`_vs_search_for_source` is monkeypatched so no OpenAI call happens; we only
assert which branches get searched and how `type` is validated/mapped.
"""

from __future__ import annotations

import os

# rag_retrieve constructs an OpenAI client at import time; the newer SDK refuses
# an empty key. Tests never hit the network (`_vs_search_for_source` is patched),
# so a dummy key is enough. Must be set before importing the module.
os.environ.setdefault("OPENAI_API_KEY", "test-key-unused")

import pytest

import tools.rag_retrieve as rr


def _hit(source_type: str = "how-to", *, id: str = "X::sec", score: float = 0.9,
         text: str = "t", file_id: str = "vs_file_1",
         filename: str | None = None) -> rr._Hit:
    return rr._Hit(
        id=id,
        source=f"documentation-{source_type}",
        text=text,
        score=score,
        file_id=file_id,
        filename=filename if filename is not None else f"{source_type}/x.md",
    )


def _record_calls(monkeypatch) -> list[dict]:
    """Record every `_vs_search_for_source` call (branch + quota + exclusions)."""
    seen: list[dict] = []

    def fake_search(query: str, source_type: str, top_k: int, exclude_ids=None):
        seen.append({"source_type": source_type, "top_k": top_k,
                     "exclude_ids": exclude_ids})
        return []

    monkeypatch.setattr(rr, "_vs_search_for_source", fake_search)
    return seen


def test_allowed_types_are_the_five_branches():
    assert set(rr.ALLOWED_TYPES) == {"language", "paradigm", "how-to", "brief", "rules"}


def test_omitted_type_searches_every_branch(monkeypatch):
    calls = _record_calls(monkeypatch)
    rr.retrieve_docs_tool("anything")
    seen = [c["source_type"] for c in calls]
    assert set(seen) == {"language", "paradigm", "how-to", "brief", "rules"}
    assert len(seen) == 5


def test_default_types_is_allowed_types():
    assert set(rr.DEFAULT_TYPES) == set(rr.ALLOWED_TYPES)


def test_guidance_branches_exclude_only_their_top_article():
    # get_guidance ships Brief.md / Rules.md in full; the detailed per-area
    # articles in the same folders must stay searchable.
    assert rr.GUIDANCE_TOP_SLUGS == {"brief": "Brief", "rules": "Rules"}
    for branch, top_slug in rr.GUIDANCE_TOP_SLUGS.items():
        assert rr._filters_for_source(branch) == {
            "type": "and",
            "filters": [
                {"type": "eq", "key": "sourceType", "value": branch},
                {"type": "ne", "key": "slug", "value": top_slug},
            ],
        }


def test_non_guidance_branches_filter_by_source_type_only():
    for branch in ("language", "paradigm", "how-to"):
        assert rr._filters_for_source(branch) == {
            "type": "eq", "key": "sourceType", "value": branch,
        }


def test_specific_type_searches_only_that_branch(monkeypatch):
    for t in rr.ALLOWED_TYPES:
        calls = _record_calls(monkeypatch)
        rr.retrieve_docs_tool("anything", type=t)
        assert [c["source_type"] for c in calls] == [t]


def test_bad_type_raises():
    with pytest.raises(ValueError):
        rr.retrieve_docs_tool("anything", type="bogus")


def test_every_allowed_type_has_a_top_k_mapping():
    for t in rr.ALLOWED_TYPES:
        assert t in rr._TYPE_TO_TOP_K


# --- chunk ids -------------------------------------------------------------


class _FakeContent:
    type = "text"

    def __init__(self, text: str):
        self.text = text


class _FakeVSHit:
    def __init__(self, attributes: dict | None, score: float = 0.7):
        self.attributes = attributes
        self.score = score
        self.content = [_FakeContent("chunk body")]
        self.file_id = "vs_file_9"
        self.filename = "how-to/x.md"


class _FakeResp:
    def __init__(self, data):
        self.data = data


def _patch_vs_search(monkeypatch, resp, captured: dict | None = None):
    def fake_search(**kwargs):
        if captured is not None:
            captured.update(kwargs)
        return resp

    monkeypatch.setattr(rr.client.vector_stores, "search", fake_search)


def test_hit_id_comes_from_section_id_attribute(monkeypatch):
    _patch_vs_search(monkeypatch, _FakeResp([
        _FakeVSHit({"section_id": "AGGR::syntax", "slug": "AGGR"}),
    ]))
    hits = rr._vs_search_for_source("q", "how-to", 3)
    assert [h.id for h in hits] == ["AGGR::syntax"]
    # not the VS file id — that stays a logging-only field
    assert hits[0].file_id == "vs_file_9" and hits[0].id != hits[0].file_id


@pytest.mark.parametrize("attrs", [
    {"slug": "AGGR"},          # attribute absent
    {"section_id": ""},        # attribute present but empty
    None,                      # no attributes at all
])
def test_hit_without_a_section_id_is_dropped_not_fatal(monkeypatch, caplog, attrs):
    # The store is external and mutable: a foreign or stale file has no
    # section_id. Dropping that one hit keeps the other branches' results,
    # which failing the whole call would throw away.
    good = _FakeVSHit({"section_id": "AGGR::syntax"})
    _patch_vs_search(monkeypatch, _FakeResp([_FakeVSHit(attrs), good]))
    with caplog.at_level("WARNING"):
        hits = rr._vs_search_for_source("q", "how-to", 3)
    assert [h.id for h in hits] == ["AGGR::syntax"]
    assert "section_id" in caplog.text


def test_doc_item_carries_the_id(monkeypatch):
    monkeypatch.setattr(rr, "_vs_search_for_source",
                        lambda q, s, k, e=None: [_hit(s, id="AGGR::syntax")])
    out = rr.retrieve_docs_tool("q", type="how-to")
    assert [d.id for d in out.docs] == ["AGGR::syntax"]


# --- exclude_ids -----------------------------------------------------------


def test_exclude_ids_adds_a_flat_nin_to_the_filter():
    assert rr._filters_for_source("how-to", ["A::x", "B::y"]) == {
        "type": "and",
        "filters": [
            {"type": "eq", "key": "sourceType", "value": "how-to"},
            {"type": "nin", "key": "section_id", "value": ["A::x", "B::y"]},
        ],
    }


def test_exclude_ids_stays_flat_next_to_the_top_slug_filter():
    f = rr._filters_for_source("rules", ["A::x"])
    assert f == {
        "type": "and",
        "filters": [
            {"type": "eq", "key": "sourceType", "value": "rules"},
            {"type": "ne", "key": "slug", "value": "Rules"},
            {"type": "nin", "key": "section_id", "value": ["A::x"]},
        ],
    }
    # one `and`, no nesting
    assert all(sub["type"] != "and" for sub in f["filters"])


def test_empty_or_none_exclude_ids_adds_no_filter():
    # `nin []` is not defined by the API contract — must not be sent at all.
    for empty in (None, [], ()):
        assert rr._filters_for_source("how-to", empty) == {
            "type": "eq", "key": "sourceType", "value": "how-to",
        }
        assert rr._filters_for_source("rules", empty) == {
            "type": "and",
            "filters": [
                {"type": "eq", "key": "sourceType", "value": "rules"},
                {"type": "ne", "key": "slug", "value": "Rules"},
            ],
        }


def test_exclude_ids_reaches_the_vector_store_filter(monkeypatch):
    captured: dict = {}
    _patch_vs_search(monkeypatch, _FakeResp([]), captured)
    rr._vs_search_for_source("q", "how-to", 3, ["A::x"])
    assert captured["filters"] == {
        "type": "and",
        "filters": [
            {"type": "eq", "key": "sourceType", "value": "how-to"},
            {"type": "nin", "key": "section_id", "value": ["A::x"]},
        ],
    }
    # The store is asked for candidates, not for the answer: the title rerank
    # picks the quota out of a wider field. The exclusions do not shrink it.
    # The branch's quota itself: with no second-stage rerank there is nothing a
    # wider candidate set could feed. (3 = the untyped per-branch quota.)
    assert captured["max_num_results"] == 3


def test_exclude_ids_is_passed_to_every_branch(monkeypatch):
    calls = _record_calls(monkeypatch)
    rr.retrieve_docs_tool("q", exclude_ids=["A::x"])
    assert len(calls) == 5
    assert all(c["exclude_ids"] == ["A::x"] for c in calls)


# --- quotas ----------------------------------------------------------------


def test_typed_quota_is_per_branch(monkeypatch):
    expected = {"rules": 7, "brief": 8}
    for t in rr.ALLOWED_TYPES:
        calls = _record_calls(monkeypatch)
        rr.retrieve_docs_tool("q", type=t)
        assert calls[0]["top_k"] == expected.get(t, 3)


def test_untyped_search_requests_three_per_branch(monkeypatch):
    calls = _record_calls(monkeypatch)
    rr.retrieve_docs_tool("q")
    assert [c["top_k"] for c in calls] == [3] * 5
    assert {c["source_type"] for c in calls} == set(rr.ALLOWED_TYPES)


def test_typed_quotas_can_return_a_whole_area_article():
    # The default table stays uniform; the typed one is sized against the corpus
    # so that asking for one area brings back all of its sections rather than a
    # majority of them chosen by the embedding.
    from settings import TOP_K, TYPED_TOP_K
    assert set(TOP_K.values()) == {3}
    assert TYPED_TOP_K["documentation-rules"] == 7
    assert TYPED_TOP_K["documentation-brief"] == 8
    assert {k: v for k, v in TYPED_TOP_K.items()
            if k not in ("documentation-rules", "documentation-brief")} == {
        k: 3 for k in TOP_K if k not in ("documentation-rules", "documentation-brief")
    }


# --- logging ---------------------------------------------------------------


def _capture_emit(monkeypatch) -> list[dict]:
    calls: list[dict] = []

    def fake_emit(event, fields, *, stream, ok=True):
        calls.append({"event": event, "fields": fields, "stream": stream, "ok": ok})

    monkeypatch.setattr(rr, "emit", fake_emit)
    return calls


def test_logs_retrieval_event_with_source_ids(monkeypatch):
    def fake_search(query, source_type, top_k, exclude_ids=None):
        return [_hit(source_type, id=f"{source_type}::s")]

    monkeypatch.setattr(rr, "_vs_search_for_source", fake_search)
    calls = _capture_emit(monkeypatch)

    out = rr.retrieve_docs_tool("how to group", type="how-to")

    assert len(out.docs) == 1  # response carries DocItem (id/source/text/score) only
    assert not hasattr(out.docs[0], "file_id")
    assert len(calls) == 1
    c = calls[0]
    assert c["event"] == "retrieve_docs" and c["stream"] == "retrieval" and c["ok"] is True
    f = c["fields"]
    assert f["type"] == "how-to" and f["n_results"] == 1 and f["top_score"] == 0.9
    assert "query" in f and "text" not in f  # no chunk text logged
    assert f["results"][0] == {"rank": 1, "source": "documentation-how-to",
                               "file_id": "vs_file_1", "filename": "how-to/x.md", "score": 0.9}


def test_logged_n_requested_follows_the_quota_actually_used(monkeypatch):
    _record_calls(monkeypatch)
    calls = _capture_emit(monkeypatch)

    rr.retrieve_docs_tool("q", type="rules")
    assert calls[-1]["fields"]["n_requested"] == 7  # typed rules quota, not 3

    rr.retrieve_docs_tool("q", type="how-to")
    assert calls[-1]["fields"]["n_requested"] == 3

    rr.retrieve_docs_tool("q")
    assert calls[-1]["fields"]["n_requested"] == 15  # 5 branches x 3, rules included


def test_log_carries_exclusion_count_not_the_ids(monkeypatch):
    _record_calls(monkeypatch)
    calls = _capture_emit(monkeypatch)

    rr.retrieve_docs_tool("q", type="how-to", exclude_ids=["A::x", "B::y", "C::z"])
    f = calls[-1]["fields"]
    assert f["n_excluded"] == 3
    assert "A::x" not in repr(f)  # ids themselves never land in the log

    rr.retrieve_docs_tool("q", type="how-to")
    assert calls[-1]["fields"]["n_excluded"] == 0


def test_failed_call_also_logs_the_exclusion_count(monkeypatch):
    calls = _capture_emit(monkeypatch)
    with pytest.raises(ValueError):
        rr.retrieve_docs_tool("q", type="bogus", exclude_ids=["A::x"])
    f = calls[0]["fields"]
    assert calls[0]["ok"] is False
    assert f["n_excluded"] == 1
    assert f["n_requested"] is None  # nothing was requested — type was rejected


def test_logs_failed_call_then_reraises(monkeypatch):
    calls = _capture_emit(monkeypatch)
    with pytest.raises(ValueError):
        rr.retrieve_docs_tool("anything", type="bogus")
    assert len(calls) == 1
    assert calls[0]["ok"] is False
    assert calls[0]["fields"]["error_class"] == "ValueError"


def test_query_is_capped_in_log(monkeypatch):
    monkeypatch.setattr(rr, "_vs_search_for_source", lambda q, s, k, e=None: [])
    monkeypatch.setattr(rr, "QUERY_LOG_MAX_CHARS", 10)
    calls = _capture_emit(monkeypatch)
    rr.retrieve_docs_tool("x" * 100)
    assert len(calls[0]["fields"]["query"]) == 10


def test_logging_failure_never_breaks_success(monkeypatch):
    monkeypatch.setattr(rr, "_vs_search_for_source",
                        lambda q, s, k, e=None: [_hit(s, score=0.5)])

    def boom(*a, **k):
        raise RuntimeError("log boom")

    monkeypatch.setattr(rr, "emit", boom)  # _log_retrieval must swallow this
    out = rr.retrieve_docs_tool("q", type="how-to")
    assert len(out.docs) == 1  # retrieval still succeeds


def test_search_error_logs_once_and_reraises_same(monkeypatch):
    sentinel = RuntimeError("vs down")

    def boom_search(q, s, k, e=None):
        raise sentinel

    monkeypatch.setattr(rr, "_vs_search_for_source", boom_search)
    calls = _capture_emit(monkeypatch)
    with pytest.raises(RuntimeError) as ei:
        rr.retrieve_docs_tool("q", type="how-to")
    assert ei.value is sentinel  # original error propagates unchanged
    assert len(calls) == 1 and calls[0]["ok"] is False
    assert calls[0]["fields"]["error_class"] == "RuntimeError"


def test_error_message_is_capped(monkeypatch):
    monkeypatch.setattr(rr, "ERROR_LOG_MAX_CHARS", 5)
    monkeypatch.setattr(rr, "_vs_search_for_source",
                        lambda q, s, k, e=None: (_ for _ in ()).throw(RuntimeError("x" * 100)))
    calls = _capture_emit(monkeypatch)
    with pytest.raises(RuntimeError):
        rr.retrieve_docs_tool("q", type="how-to")
    assert len(calls[0]["fields"]["error_message"]) == 5


def test_error_message_never_leaks_excluded_ids(monkeypatch):
    # A provider error can echo the request back, filter values included. The
    # log has no field naming the excluded ids on purpose; the error string
    # must not become one through the back door.
    records: list[dict] = []
    monkeypatch.setattr(rr, "emit", lambda name, fields, **kw: records.append(fields))

    def boom(*a, **kw):
        raise RuntimeError("400 bad filter: {'nin': ['AGGR::syntax', 'GROUP::examples']}")

    monkeypatch.setattr(rr, "_vs_search_for_source", boom)
    with pytest.raises(RuntimeError):
        rr.retrieve_docs_tool("q", type="how-to",
                              exclude_ids=["AGGR::syntax", "GROUP::examples"])

    assert len(records) == 1
    blob = str(records[0])
    assert "AGGR::syntax" not in blob
    assert "GROUP::examples" not in blob
    assert "<excluded-id>" in records[0]["error_message"]
    assert records[0]["n_excluded"] == 2

# --- title rerank -----------------------------------------------------------


def _vshit(section_id: str, heading_path: str, score: float, slug: str = ""):
    return _FakeVSHit(
        {"section_id": section_id, "heading_path": heading_path, "slug": slug},
        score=score,
    )