"""Tests for tools/feedback.py — recording, redaction, dedup, caps, kill switch."""
from __future__ import annotations

import tools.feedback as re_mod
from tools.feedback import (
    FeedbackReport,
    EvalError,
    Expectation,
    Recommendation,
    RetrieveQuery,
    report_feedback_tool,
)


def _capture_emit(monkeypatch) -> list[dict]:
    calls: list[dict] = []

    def fake_emit(event, fields, *, stream, ok=True):
        calls.append({"event": event, "fields": fields, "stream": stream, "ok": ok})

    monkeypatch.setattr(re_mod, "emit", fake_emit)
    return calls


def _report(**over) -> FeedbackReport:
    base = dict(
        agent_journey_id="11111111-1111-1111-1111-111111111111",
        signal_type="doc-gap",
        problem_summary="aggregate orders by customer",
        recommendation=Recommendation(primary_target="how-to", suggested_change="add GROUP example"),
        eval_errors=[EvalError(message="Unknown property foo", phase="semantic")],
        retrieve_queries=[RetrieveQuery(query="group sum", usefulness="incomplete")],
        final_outcome="fixed",
    )
    base.update(over)
    return FeedbackReport(**base)


def test_records_and_returns_dedup(monkeypatch):
    calls = _capture_emit(monkeypatch)
    out = report_feedback_tool(_report())
    assert out.status == "recorded"
    assert out.report_id.startswith("rpt_") and out.dedup_fingerprint
    assert len(calls) == 1
    c = calls[0]
    assert c["event"] == "report_feedback" and c["stream"] == "reports" and c["ok"] is True
    assert c["fields"]["report_id"] == out.report_id
    assert c["fields"]["dedup_fingerprint"] == out.dedup_fingerprint


def test_redacts_pii_in_logged_payload(monkeypatch):
    calls = _capture_emit(monkeypatch)
    rep = _report(
        problem_summary="contact bob@acme.com about 10.0.0.5",
        eval_errors=[EvalError(message="key sk-ABCDEF0123456789 failed at /home/alex/proj/x.lsf")],
    )
    report_feedback_tool(rep)
    f = calls[0]["fields"]
    blob = str(f)
    assert "bob@acme.com" not in blob and "<redacted-email>" in blob
    assert "10.0.0.5" not in blob
    assert "sk-ABCDEF0123456789" not in blob
    assert "/home/alex/proj/x.lsf" not in blob
    # agent_journey_id (a uuid) must NOT be mangled
    assert f["agent_journey_id"] == "11111111-1111-1111-1111-111111111111"


def test_dedup_is_deterministic(monkeypatch):
    _capture_emit(monkeypatch)
    a = report_feedback_tool(_report())
    b = report_feedback_tool(_report())
    assert a.dedup_fingerprint == b.dedup_fingerprint
    assert a.report_id != b.report_id  # ids are unique per submission


def test_disabled_switch(monkeypatch):
    monkeypatch.setattr(re_mod, "FEEDBACK_ENABLED", False)
    calls = _capture_emit(monkeypatch)
    out = report_feedback_tool(_report())
    assert out.status == "disabled"
    assert calls == []  # nothing stored


def test_rejects_oversized_and_logs_minimal(monkeypatch):
    monkeypatch.setattr(re_mod, "REPORT_MAX_EVAL_ERRORS", 2)
    calls = _capture_emit(monkeypatch)
    rep = _report(eval_errors=[EvalError(message=f"e{i}") for i in range(5)])
    out = report_feedback_tool(rep)
    assert out.status == "rejected" and "eval_errors" in (out.detail or "")
    assert len(calls) == 1 and calls[0]["ok"] is False
    # minimal rejected event carries no user payload
    assert "problem_summary" not in calls[0]["fields"]
    assert calls[0]["fields"]["reason"] == out.detail


def test_rejects_large_code_excerpt(monkeypatch):
    monkeypatch.setattr(re_mod, "REPORT_CODE_EXCERPT_MAX_CHARS", 10)
    _capture_emit(monkeypatch)
    rep = _report(eval_errors=[EvalError(message="oops", code_excerpt="x" * 50)])
    out = report_feedback_tool(rep)
    assert out.status == "rejected" and "code_excerpt" in (out.detail or "")


def test_expectation_mismatch_signal_records(monkeypatch):
    calls = _capture_emit(monkeypatch)
    rep = _report(
        signal_type="expectation-mismatch",
        eval_errors=[],
        expectation=Expectation(expected="GROUP keeps all rows",
                                actual="GROUP aggregates to one row per key"),
        recommendation=Recommendation(primary_target="paradigm",
                                      suggested_change="clarify GROUP semantics"),
    )
    out = report_feedback_tool(rep)
    assert out.status == "recorded"
    f = calls[0]["fields"]
    assert f["signal_type"] == "expectation-mismatch"
    assert f["expectation"]["expected"].startswith("GROUP keeps")


def test_eval_error_message_target_allowed(monkeypatch):
    _capture_emit(monkeypatch)
    rep = _report(
        signal_type="unclear-error",
        recommendation=Recommendation(primary_target="eval-error-message",
                                      suggested_change="say which property is unknown"),
    )
    assert report_feedback_tool(rep).status == "recorded"


def test_client_dedup_hint_is_redacted(monkeypatch):
    calls = _capture_emit(monkeypatch)
    report_feedback_tool(_report(client_dedup_hint="ping bob@acme.com"))
    blob = str(calls[0]["fields"])
    assert "bob@acme.com" not in blob and "<redacted-email>" in blob


def test_tool_context_drops_unknown_keys(monkeypatch):
    calls = _capture_emit(monkeypatch)
    rep = _report(tool_context={"eval_server_kind": "demo", "customer_acme": "secret-name"})
    report_feedback_tool(rep)
    tc = calls[0]["fields"]["tool_context"]
    assert tc == {"eval_server_kind": "demo"}  # unknown key dropped (extra='ignore')


def test_redacts_nested_query_sources_and_suggestion(monkeypatch):
    calls = _capture_emit(monkeypatch)
    rep = _report(
        retrieve_queries=[RetrieveQuery(query="see /home/alex/x.lsf", returned_sources=["10.1.2.3"])],
        recommendation=Recommendation(primary_target="how-to", suggested_change="mail a@b.com"),
    )
    report_feedback_tool(rep)
    blob = str(calls[0]["fields"])
    assert "/home/alex/x.lsf" not in blob and "10.1.2.3" not in blob and "a@b.com" not in blob


def test_rejects_too_many_queries(monkeypatch):
    monkeypatch.setattr(re_mod, "REPORT_MAX_QUERIES", 2)
    calls = _capture_emit(monkeypatch)
    out = report_feedback_tool(_report(retrieve_queries=[RetrieveQuery(query=f"q{i}") for i in range(5)]))
    assert out.status == "rejected" and "retrieve_queries" in (out.detail or "")
    assert "problem_summary" not in calls[0]["fields"]  # no payload leak on reject


def test_byte_size_cap_counts_bytes(monkeypatch):
    monkeypatch.setattr(re_mod, "REPORT_MAX_TOTAL_CHARS", 80)
    _capture_emit(monkeypatch)
    out = report_feedback_tool(_report(problem_summary="嗨" * 40))  # 40 chars, 120 bytes
    assert out.status == "rejected" and "bytes" in (out.detail or "")


def test_emit_failure_does_not_raise(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("emit down")

    monkeypatch.setattr(re_mod, "emit", boom)
    out = report_feedback_tool(_report())  # must not propagate
    assert out.status == "rejected" and out.detail == "processing error"


def test_oversized_field_rejected_at_parse():
    import pydantic
    import pytest
    with pytest.raises(pydantic.ValidationError):
        _report(problem_summary="x" * 5000)  # exceeds max_length


def test_dedup_distinguishes_signal_type(monkeypatch):
    _capture_emit(monkeypatch)
    a = report_feedback_tool(_report(signal_type="doc-gap"))
    b = report_feedback_tool(_report(signal_type="unclear-error"))
    # same summary/error/target but different signal_type => different cluster
    assert a.dedup_fingerprint != b.dedup_fingerprint
