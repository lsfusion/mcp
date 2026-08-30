"""Tests for tools/triage_digest.py — clustering, gaps, operational, robustness."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import tools.triage_digest as td


def _write(dirpath, name, rows):
    p = dirpath / name
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def _retr(query, n_results=3, top_score=0.8, ok=True, ts="2026-05-30T10:00:00.000Z"):
    return {"schema_version": 1, "event": "retrieve_docs", "ts": ts, "ok": ok,
            "query": query, "type": "how-to", "n_requested": 3, "n_results": n_results,
            "top_score": top_score, "latency_ms": 100, "results": []}


def _rep(fp, signal="doc-gap", target="how-to", outcome="fixed", journey="j1",
         summary="group orders", ok=True, ts="2026-05-30T10:00:00.000Z"):
    return {"schema_version": 1, "event": "report_feedback", "ts": ts, "ok": ok,
            "agent_journey_id": journey, "signal_type": signal, "problem_summary": summary,
            "recommendation": {"primary_target": target, "suggested_change": "add example"},
            "final_outcome": outcome, "report_id": "rpt_x", "dedup_fingerprint": fp}


def test_feedback_clusters_grouped_by_fingerprint(tmp_path):
    _write(tmp_path, "reports-20260530.jsonl", [
        _rep("AAAA", journey="j1"), _rep("AAAA", journey="j2"), _rep("BBBB", journey="j3"),
    ])
    _write(tmp_path, "retrieval-20260530.jsonl", [_retr("x")])
    out = td.build_digest(str(tmp_path), None, 0.5, 15)
    assert "Feedback clusters (3 reports)" in out
    assert "**AAAA** ×2 (2 journeys)" in out
    assert "**BBBB** ×1" in out
    # AAAA (bigger) listed before BBBB
    assert out.index("**AAAA**") < out.index("**BBBB**")


def test_outcome_unresolved_count(tmp_path):
    _write(tmp_path, "reports-20260530.jsonl", [
        _rep("A", outcome="fixed"), _rep("B", outcome="abandoned"), _rep("C", outcome="not_fixed"),
    ])
    out = td.build_digest(str(tmp_path), None, 0.5, 15)
    assert "2/3 (67%) not cleanly fixed" in out


def test_retrieval_gaps_empty_low_and_frequent(tmp_path):
    _write(tmp_path, "retrieval-20260530.jsonl", [
        _retr("missing topic", n_results=0, top_score=None),
        _retr("weak topic", top_score=0.2),
        _retr("weak topic", top_score=0.3),
        _retr("popular", top_score=0.9), _retr("popular", top_score=0.9), _retr("popular", top_score=0.9),
    ])
    out = td.build_digest(str(tmp_path), None, 0.5, 15)
    assert "missing topic" in out and "Empty-result queries" in out
    assert "weak topic" in out  # low-score, aggregated ×2
    assert "×3 — popular" in out  # demand signal


def test_operational_rejections_and_errors(tmp_path):
    _write(tmp_path, "reports-20260530.jsonl", [
        _rep("A"),
        {"event": "report_feedback", "ok": False, "report_id": "r", "reason": "too many eval_errors (>50)", "payload_size": 9},
    ])
    _write(tmp_path, "retrieval-20260530.jsonl", [
        _retr("q"),
        {"event": "event_log_error", "ok": False, "schema_version": 1, "ts": "2026-05-30T10:00:00.000Z"},
    ])
    out = td.build_digest(str(tmp_path), None, 0.5, 15)
    assert "Rejected feedback submissions: 1" in out
    assert "too many eval_errors (>50)" in out
    assert "event_log_error lines (persistence failures): 1" in out


def test_torn_line_is_skipped(tmp_path):
    p = tmp_path / "reports-20260530.jsonl"
    p.write_text(json.dumps(_rep("A")) + "\n{ broken json\n" + json.dumps(_rep("A")) + "\n",
                 encoding="utf-8")
    _write(tmp_path, "retrieval-20260530.jsonl", [_retr("q")])
    out = td.build_digest(str(tmp_path), None, 0.5, 15)
    assert "Feedback clusters (2 reports)" in out  # broken middle line ignored


def test_empty_dir_is_graceful(tmp_path):
    out = td.build_digest(str(tmp_path), None, 0.5, 15)
    assert "No recorded feedback." in out and "No retrieval activity." in out


def test_days_window_filters(tmp_path):
    """The window is relative to NOW, so the "recent" row has to be written
    relative to now as well. A fixed date passes on the day it is written and
    fails a month later, which is how this test spent a summer red."""
    recent = datetime.now(timezone.utc) - timedelta(days=1)
    _write(tmp_path, f"reports-{recent:%Y%m%d}.jsonl", [
        _rep("RECENT", ts=recent.isoformat().replace("+00:00", "Z")),
        _rep("OLD", ts="2020-01-01T00:00:00.000Z"),
    ])
    _write(tmp_path, f"retrieval-{recent:%Y%m%d}.jsonl", [_retr("q")])
    out_all = td.build_digest(str(tmp_path), None, 0.5, 15)
    assert "Feedback clusters (2 reports)" in out_all
    out_recent = td.build_digest(str(tmp_path), 30, 0.5, 15)  # 30d window drops the 2020 row
    assert "OLD" not in out_recent
    assert "Feedback clusters (1 reports)" in out_recent
