"""Tests for tools/event_log.py — envelope shape + best-effort contract."""
from __future__ import annotations

import json

import tools.event_log as el


def test_emit_writes_envelope_to_stderr(capsys):
    el.emit("retrieve_docs", {"n_results": 2}, stream="retrieval", ok=True)
    err = capsys.readouterr().err.strip()
    rec = json.loads(err)  # one valid JSON object per line
    assert rec["event"] == "retrieve_docs"
    assert rec["schema_version"] == el.LOG_SCHEMA_VERSION
    assert rec["ok"] is True
    assert rec["n_results"] == 2
    assert rec["ts"].endswith("Z")
    assert "server_version" in rec


def test_emit_never_raises_on_unserializable(capsys):
    # A non-JSON-serializable field must not blow up the caller (best-effort).
    el.emit("retrieve_docs", {"bad": object()}, stream="retrieval")
    assert capsys.readouterr().err == ""  # nothing emitted, no traceback


def test_ts_is_utc_millis():
    ts = el._utc_now_iso()
    assert ts.endswith("Z") and "T" in ts
    # milliseconds: <date>T<hh:mm:ss>.<3 digits>Z
    assert len(ts.split(".")[-1]) == 4  # "789Z"


def test_emit_writes_nothing_to_stdout(capsys):
    el.emit("retrieve_docs", {"n_results": 0}, stream="retrieval")
    cap = capsys.readouterr()
    assert cap.out == ""  # stdout (MCP STDIO channel) stays clean
    assert cap.err.strip() != ""


def test_log_dir_failure_is_nonfatal_and_json(capsys, monkeypatch):
    monkeypatch.setattr(el, "LOG_DIR", "/nonexistent/definitely/nope")
    el.emit("retrieve_docs", {"n_results": 0}, stream="retrieval")  # must not raise
    lines = [l for l in capsys.readouterr().err.splitlines() if l.strip()]
    recs = [json.loads(l) for l in lines]  # every stderr line stays valid JSON
    assert any(r["event"] == "event_log_error" and r["ok"] is False for r in recs)
