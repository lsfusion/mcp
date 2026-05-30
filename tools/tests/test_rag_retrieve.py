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


def _record_branches(monkeypatch) -> list[str]:
    seen: list[str] = []

    def fake_search(query: str, source_type: str, top_k: int):
        seen.append(source_type)
        return []

    monkeypatch.setattr(rr, "_vs_search_for_source", fake_search)
    return seen


def test_allowed_types_are_the_five_branches():
    assert set(rr.ALLOWED_TYPES) == {"language", "paradigm", "how-to", "brief", "rules"}


def test_omitted_type_searches_all_five_branches(monkeypatch):
    seen = _record_branches(monkeypatch)
    rr.retrieve_docs_tool("anything")
    assert set(seen) == set(rr.ALLOWED_TYPES)
    assert len(seen) == 5


def test_specific_type_searches_only_that_branch(monkeypatch):
    for t in rr.ALLOWED_TYPES:
        seen = _record_branches(monkeypatch)
        rr.retrieve_docs_tool("anything", type=t)
        assert seen == [t]


def test_bad_type_raises():
    with pytest.raises(ValueError):
        rr.retrieve_docs_tool("anything", type="bogus")


def test_every_allowed_type_has_a_top_k_mapping():
    for t in rr.ALLOWED_TYPES:
        assert t in rr._TYPE_TO_TOP_K


def _capture_emit(monkeypatch) -> list[dict]:
    calls: list[dict] = []

    def fake_emit(event, fields, *, stream, ok=True):
        calls.append({"event": event, "fields": fields, "stream": stream, "ok": ok})

    monkeypatch.setattr(rr, "emit", fake_emit)
    return calls


def test_logs_retrieval_event_with_source_ids(monkeypatch):
    def fake_search(query, source_type, top_k):
        return [rr._Hit(source=f"documentation-{source_type}", text="t",
                        score=0.9, file_id="vs_file_1", filename=f"{source_type}/x.md")]

    monkeypatch.setattr(rr, "_vs_search_for_source", fake_search)
    calls = _capture_emit(monkeypatch)

    out = rr.retrieve_docs_tool("how to group", type="how-to")

    assert len(out.docs) == 1  # response carries DocItem (source/text/score) only
    assert not hasattr(out.docs[0], "file_id")
    assert len(calls) == 1
    c = calls[0]
    assert c["event"] == "retrieve_docs" and c["stream"] == "retrieval" and c["ok"] is True
    f = c["fields"]
    assert f["type"] == "how-to" and f["n_results"] == 1 and f["top_score"] == 0.9
    assert "query" in f and "text" not in f  # no chunk text logged
    assert f["results"][0] == {"rank": 1, "source": "documentation-how-to",
                               "file_id": "vs_file_1", "filename": "how-to/x.md", "score": 0.9}


def test_logs_failed_call_then_reraises(monkeypatch):
    calls = _capture_emit(monkeypatch)
    with pytest.raises(ValueError):
        rr.retrieve_docs_tool("anything", type="bogus")
    assert len(calls) == 1
    assert calls[0]["ok"] is False
    assert calls[0]["fields"]["error_class"] == "ValueError"


def test_query_is_capped_in_log(monkeypatch):
    monkeypatch.setattr(rr, "_vs_search_for_source", lambda q, s, k: [])
    monkeypatch.setattr(rr, "QUERY_LOG_MAX_CHARS", 10)
    calls = _capture_emit(monkeypatch)
    rr.retrieve_docs_tool("x" * 100)
    assert len(calls[0]["fields"]["query"]) == 10


def test_logging_failure_never_breaks_success(monkeypatch):
    monkeypatch.setattr(rr, "_vs_search_for_source",
                        lambda q, s, k: [rr._Hit("documentation-how-to", "t", 0.5, "f", "x.md")])

    def boom(*a, **k):
        raise RuntimeError("log boom")

    monkeypatch.setattr(rr, "emit", boom)  # _log_retrieval must swallow this
    out = rr.retrieve_docs_tool("q", type="how-to")
    assert len(out.docs) == 1  # retrieval still succeeds


def test_search_error_logs_once_and_reraises_same(monkeypatch):
    sentinel = RuntimeError("vs down")

    def boom_search(q, s, k):
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
                        lambda q, s, k: (_ for _ in ()).throw(RuntimeError("x" * 100)))
    calls = _capture_emit(monkeypatch)
    with pytest.raises(RuntimeError):
        rr.retrieve_docs_tool("q", type="how-to")
    assert len(calls[0]["fields"]["error_message"]) == 5
