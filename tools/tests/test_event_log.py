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


def test_log_dir_persists_dated_jsonl_file(tmp_path, monkeypatch):
    # End-to-end: with LOG_DIR set, emit actually appends a parseable line to a
    # dated <stream>-YYYYMMDD.jsonl file (the durable sink, no mocks).
    import datetime
    import glob
    monkeypatch.setattr(el, "LOG_DIR", str(tmp_path))
    el.emit("retrieve_docs", {"n_results": 2}, stream="retrieval", ok=True)
    el.emit("report_feedback", {"report_id": "rpt_x"}, stream="reports", ok=True)

    day = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
    rfiles = glob.glob(str(tmp_path / f"retrieval-{day}.jsonl"))
    pfiles = glob.glob(str(tmp_path / f"reports-{day}.jsonl"))
    assert rfiles and pfiles  # both streams land in their own dated file
    rec = json.loads(open(rfiles[0], encoding="utf-8").read().strip())
    assert rec["event"] == "retrieve_docs" and rec["n_results"] == 2
    assert json.loads(open(pfiles[0], encoding="utf-8").read().strip())["report_id"] == "rpt_x"


def test_log_dir_appends_not_truncates(tmp_path, monkeypatch):
    monkeypatch.setattr(el, "LOG_DIR", str(tmp_path))
    for i in range(3):
        el.emit("retrieve_docs", {"n": i}, stream="retrieval")
    import datetime
    day = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
    lines = [l for l in open(tmp_path / f"retrieval-{day}.jsonl", encoding="utf-8") if l.strip()]
    assert len(lines) == 3  # appended, one JSON object per line
    assert [json.loads(l)["n"] for l in lines] == [0, 1, 2]
