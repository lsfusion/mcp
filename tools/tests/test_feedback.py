"""Tests for tools/feedback.py — storing, redaction, dedup, caps, kill switch."""
from __future__ import annotations

import json
import os
import pathlib
from datetime import datetime, timezone

import pytest

import tools.feedback as re_mod
from tools.feedback import (
    STORE_FAILED,
    FeedbackReport,
    EvalError,
    Expectation,
    Recommendation,
    RetrieveQuery,
    report_feedback_tool,
)


@pytest.fixture(autouse=True)
def inbox(tmp_path, monkeypatch) -> pathlib.Path:
    # The reports dir does not exist yet: the tool must create it.
    monkeypatch.setattr(re_mod, "REPORTS_DIR", str(tmp_path / "reports"))
    # Outside a request caller_fields() is {} anyway; pin it so no test depends on that.
    monkeypatch.setattr(re_mod, "caller_fields", lambda: {})
    return tmp_path / "reports" / "inbox"


def _files(inbox: pathlib.Path) -> list[pathlib.Path]:
    """Everything in the inbox, temp files included."""
    return sorted(inbox.iterdir()) if inbox.exists() else []


def _stored(inbox: pathlib.Path) -> dict:
    files = _files(inbox)
    assert len(files) == 1, files
    return json.loads(files[0].read_text(encoding="utf-8"))


def _capture_emit(monkeypatch) -> list[dict]:
    calls: list[dict] = []

    def fake_emit(event, fields, *, stream, ok=True):
        calls.append({"event": event, "fields": fields, "stream": stream, "ok": ok})

    monkeypatch.setattr(re_mod, "emit", fake_emit)
    return calls


def _at(*args):
    return lambda: datetime(*args, tzinfo=timezone.utc)


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


# --- storing ------------------------------------------------------------------

def test_stores_one_file_named_by_time_and_id(inbox, monkeypatch):
    _capture_emit(monkeypatch)
    monkeypatch.setattr(re_mod, "_utc_now", _at(2026, 11, 19, 10, 15, 0, 123456))
    out = report_feedback_tool(_report(agent="claude-code, claude-opus-4-5"))
    assert out.status == "recorded"
    assert out.report_id.startswith("rpt_") and out.dedup_fingerprint
    assert [f.name for f in _files(inbox)] == [f"20261119T101500-{out.report_id}.json"]

    rec = _stored(inbox)
    assert rec["report_id"] == out.report_id
    assert rec["ts"] == "2026-11-19T10:15:00.123Z"
    assert rec["schema_version"] == re_mod.LOG_SCHEMA_VERSION
    assert rec["server_version"] == re_mod.SERVER_VERSION
    assert rec["dedup_fingerprint"] == out.dedup_fingerprint
    # ...and the report itself, everything the worker needs for the issue.
    assert rec["signal_type"] == "doc-gap"
    assert rec["problem_summary"] == "aggregate orders by customer"
    assert rec["recommendation"]["primary_target"] == "how-to"
    assert rec["eval_errors"][0]["message"] == "Unknown property foo"
    assert rec["retrieve_queries"][0]["query"] == "group sum"
    assert rec["agent"] == "claude-code, claude-opus-4-5"
    # Unknown client => the keys are missing, not empty.
    assert "client" not in rec and "client_version" not in rec


def test_file_names_sort_in_time_order(inbox, monkeypatch):
    _capture_emit(monkeypatch)
    monkeypatch.setattr(re_mod, "_utc_now", _at(2026, 11, 19, 23, 59, 59))
    first = report_feedback_tool(_report())
    monkeypatch.setattr(re_mod, "_utc_now", _at(2026, 11, 20, 0, 0, 0))
    second = report_feedback_tool(_report())
    # The ids are random; only the timestamp prefix can put these in order.
    assert [f.name.split("-", 1)[1] for f in _files(inbox)] == [
        f"{first.report_id}.json", f"{second.report_id}.json"]


def test_writes_a_complete_temp_file_then_renames_it(inbox, monkeypatch):
    _capture_emit(monkeypatch)
    real_replace = os.replace
    seen = []

    def spy(src, dst):
        src, dst = pathlib.Path(src), pathlib.Path(dst)
        seen.append((src, dst, dst.exists(), json.loads(src.read_text(encoding="utf-8"))))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    out = report_feedback_tool(_report())
    assert out.status == "recorded"
    (src, dst, existed, content), = seen
    # Same folder (the rename cannot cross filesystems), and a name a worker
    # reading *.json never picks up.
    assert src.parent == dst.parent == inbox
    assert src.name.startswith(".") and src.name.endswith(".tmp")
    # The final name appears only through the rename, and the temp file was
    # already whole when it happened.
    assert not existed
    assert content["report_id"] == out.report_id
    assert [f.name for f in _files(inbox)] == [dst.name]


def test_recorded_only_once_the_file_is_there(inbox, monkeypatch):
    in_inbox_when_logged = []
    monkeypatch.setattr(re_mod, "emit", lambda *a, **k: in_inbox_when_logged.append(
        [f.name for f in _files(inbox)]))
    out = report_feedback_tool(_report())
    assert out.status == "recorded"
    assert len(in_inbox_when_logged) == 1
    assert in_inbox_when_logged[0] == [f.name for f in _files(inbox)]
    assert in_inbox_when_logged[0][0].endswith(f"-{out.report_id}.json")


def test_rejected_when_the_write_fails_and_nothing_is_left(inbox, monkeypatch):
    calls = _capture_emit(monkeypatch)

    def fail(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", fail)
    out = report_feedback_tool(_report())
    assert out.status == "rejected" and out.detail == STORE_FAILED
    assert out.dedup_fingerprint is None
    assert _files(inbox) == []  # the temp file went too
    assert len(calls) == 1
    c = calls[0]
    assert c["event"] == "report_feedback" and c["stream"] == "reports" and c["ok"] is False
    assert c["fields"]["reason"] == STORE_FAILED and c["fields"]["cause"] == "OSError"
    assert "problem_summary" not in c["fields"]


def test_rejected_when_the_inbox_cannot_be_created(tmp_path, monkeypatch):
    _capture_emit(monkeypatch)
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    monkeypatch.setattr(re_mod, "REPORTS_DIR", str(blocker))
    out = report_feedback_tool(_report())
    assert out.status == "rejected" and out.detail == STORE_FAILED


def test_rejected_without_a_reports_dir(monkeypatch):
    monkeypatch.setattr(re_mod, "REPORTS_DIR", "")
    calls = _capture_emit(monkeypatch)
    out = report_feedback_tool(_report())
    assert out.status == "rejected" and out.detail == STORE_FAILED
    assert calls[0]["ok"] is False
    assert calls[0]["fields"]["cause"] == "no reports dir configured"
    assert "problem_summary" not in calls[0]["fields"]


def test_stored_event_is_short_and_carries_no_content(inbox, monkeypatch):
    calls = _capture_emit(monkeypatch)
    out = report_feedback_tool(_report())
    assert len(calls) == 1
    c = calls[0]
    assert c["event"] == "report_stored" and c["stream"] == "reports" and c["ok"] is True
    assert set(c["fields"]) == {"report_id", "signal_type", "payload_size"}
    assert c["fields"]["report_id"] == out.report_id
    assert c["fields"]["signal_type"] == "doc-gap"
    assert c["fields"]["payload_size"] > 0
    blob = json.dumps(c["fields"])
    for text in ("aggregate orders", "Unknown property", "group sum", "add GROUP"):
        assert text not in blob


def test_client_fields_come_from_the_handshake(inbox, monkeypatch):
    _capture_emit(monkeypatch)
    monkeypatch.setattr(re_mod, "caller_fields", lambda: {
        "client": "claude-code", "client_version": "2.1.3",
        "user_agent": "node", "declared_model": "opus"})
    report_feedback_tool(_report())
    rec = _stored(inbox)
    assert rec["client"] == "claude-code" and rec["client_version"] == "2.1.3"
    assert "user_agent" not in rec and "declared_model" not in rec


def test_a_client_without_a_version_stores_just_the_name(inbox, monkeypatch):
    _capture_emit(monkeypatch)
    monkeypatch.setattr(re_mod, "caller_fields", lambda: {"client": "cursor"})
    report_feedback_tool(_report())
    rec = _stored(inbox)
    assert rec["client"] == "cursor" and "client_version" not in rec


def test_non_ascii_text_is_stored_as_utf8(inbox, monkeypatch):
    _capture_emit(monkeypatch)
    report_feedback_tool(_report(problem_summary="згрупаваць заказы па кліенце"))
    assert _stored(inbox)["problem_summary"] == "згрупаваць заказы па кліенце"


def test_log_failure_after_storing_still_records(inbox, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("emit down")

    monkeypatch.setattr(re_mod, "emit", boom)
    out = report_feedback_tool(_report())  # must not propagate
    # The file is in the inbox: saying anything but `recorded` would get it sent twice.
    assert out.status == "recorded"
    assert _stored(inbox)["report_id"] == out.report_id


def test_log_failure_while_rejecting_does_not_raise(inbox, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("emit down")

    monkeypatch.setattr(re_mod, "emit", boom)
    monkeypatch.setattr(re_mod, "REPORT_MAX_EVAL_ERRORS", 0)
    out = report_feedback_tool(_report())
    assert out.status == "rejected" and out.detail == "processing error"
    assert _files(inbox) == []


# --- redaction ----------------------------------------------------------------

def test_redacts_pii_in_stored_report(inbox, monkeypatch):
    _capture_emit(monkeypatch)
    rep = _report(
        problem_summary="contact bob@acme.com about 10.0.0.5",
        eval_errors=[EvalError(message="key sk-ABCDEF0123456789 failed at /home/alex/proj/x.lsf")],
    )
    report_feedback_tool(rep)
    rec = _stored(inbox)
    blob = json.dumps(rec, ensure_ascii=False)
    assert "bob@acme.com" not in blob and "<redacted-email>" in blob
    assert "10.0.0.5" not in blob
    assert "sk-ABCDEF0123456789" not in blob
    assert "/home/alex/proj/x.lsf" not in blob
    # agent_journey_id (a uuid) must NOT be mangled
    assert rec["agent_journey_id"] == "11111111-1111-1111-1111-111111111111"


def test_client_dedup_hint_is_redacted(inbox, monkeypatch):
    _capture_emit(monkeypatch)
    report_feedback_tool(_report(client_dedup_hint="ping bob@acme.com"))
    blob = json.dumps(_stored(inbox))
    assert "bob@acme.com" not in blob and "<redacted-email>" in blob


def test_tool_context_drops_unknown_keys(inbox, monkeypatch):
    _capture_emit(monkeypatch)
    rep = _report(tool_context={"eval_server_kind": "demo", "customer_acme": "secret-name"})
    report_feedback_tool(rep)
    assert _stored(inbox)["tool_context"] == {"eval_server_kind": "demo"}  # extra='ignore'


def test_redacts_nested_query_sources_and_suggestion(inbox, monkeypatch):
    _capture_emit(monkeypatch)
    rep = _report(
        retrieve_queries=[RetrieveQuery(query="see /home/alex/x.lsf", returned_sources=["10.1.2.3"])],
        recommendation=Recommendation(primary_target="how-to", suggested_change="mail a@b.com"),
    )
    report_feedback_tool(rep)
    blob = json.dumps(_stored(inbox))
    assert "/home/alex/x.lsf" not in blob and "10.1.2.3" not in blob and "a@b.com" not in blob


# --- signals and dedup ----------------------------------------------------------

def test_dedup_is_deterministic(monkeypatch):
    _capture_emit(monkeypatch)
    a = report_feedback_tool(_report())
    b = report_feedback_tool(_report())
    assert a.dedup_fingerprint == b.dedup_fingerprint
    assert a.report_id != b.report_id  # ids are unique per submission


def test_dedup_distinguishes_signal_type(monkeypatch):
    _capture_emit(monkeypatch)
    a = report_feedback_tool(_report(signal_type="doc-gap"))
    b = report_feedback_tool(_report(signal_type="unclear-error"))
    # same summary/error/target but different signal_type => different cluster
    assert a.dedup_fingerprint != b.dedup_fingerprint


def test_expectation_mismatch_signal_records(inbox, monkeypatch):
    _capture_emit(monkeypatch)
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
    rec = _stored(inbox)
    assert rec["signal_type"] == "expectation-mismatch"
    assert rec["expectation"]["expected"].startswith("GROUP keeps")


def test_eval_error_message_target_allowed(monkeypatch):
    _capture_emit(monkeypatch)
    rep = _report(
        signal_type="unclear-error",
        recommendation=Recommendation(primary_target="eval-error-message",
                                      suggested_change="say which property is unknown"),
    )
    assert report_feedback_tool(rep).status == "recorded"


# --- kill switch and caps ---------------------------------------------------------

def test_disabled_switch(inbox, monkeypatch):
    monkeypatch.setattr(re_mod, "FEEDBACK_ENABLED", False)
    calls = _capture_emit(monkeypatch)
    out = report_feedback_tool(_report())
    assert out.status == "disabled"
    assert calls == [] and _files(inbox) == []  # nothing stored


def test_rejects_oversized_and_logs_minimal(inbox, monkeypatch):
    monkeypatch.setattr(re_mod, "REPORT_MAX_EVAL_ERRORS", 2)
    calls = _capture_emit(monkeypatch)
    rep = _report(eval_errors=[EvalError(message=f"e{i}") for i in range(5)])
    out = report_feedback_tool(rep)
    assert out.status == "rejected" and "eval_errors" in (out.detail or "")
    assert len(calls) == 1 and calls[0]["ok"] is False
    # minimal rejected event carries no user payload
    assert "problem_summary" not in calls[0]["fields"]
    assert calls[0]["fields"]["reason"] == out.detail
    assert _files(inbox) == []


def test_rejects_large_code_excerpt(monkeypatch):
    monkeypatch.setattr(re_mod, "REPORT_CODE_EXCERPT_MAX_CHARS", 10)
    _capture_emit(monkeypatch)
    rep = _report(eval_errors=[EvalError(message="oops", code_excerpt="x" * 50)])
    out = report_feedback_tool(rep)
    assert out.status == "rejected" and "code_excerpt" in (out.detail or "")


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


def test_oversized_field_rejected_at_parse():
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        _report(problem_summary="x" * 5000)  # exceeds max_length
