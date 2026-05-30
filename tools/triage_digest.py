"""Phase C triage digest — turn the JSONL event sinks into a human review doc.

Offline, dependency-free. Reads the `retrieval-*.jsonl` and `reports-*.jsonl`
files written by event_log (LOG_DIR, e.g. /opt/stack/mcp-data/logs) and prints a
Markdown digest that CLUSTERS the signals so a human can decide doc edits:

  * Feedback clusters — report_feedback grouped by the server-computed
    `dedup_fingerprint`, with signal_type / primary_target / outcome breakdowns.
  * Retrieval coverage gaps — queries that returned nothing or scored low, and
    the most frequent queries (demand signal).
  * Operational health — rejected reports (by reason) and event_log_error lines.

A report is a RECOMMENDATION, not a decision: this digest never edits docs and
never opens issues (MCP-FEEDBACK-PLAN.md Phase C). A human reads it and edits
How-to / Rules / Brief or files a bug.

Usage:
    python -m tools.triage_digest --log-dir /opt/stack/mcp-data/logs [--days N]
        [--low-score 0.5] [--top 15] [--out digest.md]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone


def _read_jsonl(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # skip a torn/partial line, keep going
        except OSError:
            continue
    return rows


def _within_days(rows: list[dict], days: int | None) -> list[dict]:
    if not days:
        return rows
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    out = []
    for r in rows:
        ts = r.get("ts", "")
        try:
            t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            out.append(r)  # undated → keep
            continue
        if t >= cutoff:
            out.append(r)
    return out


def _norm(s) -> str:
    return " ".join(str(s or "").lower().split())


def _pct(n: int, total: int) -> str:
    return f"{(100 * n / total):.0f}%" if total else "0%"


def load_events(log_dir: str, days: int | None):
    retrieval = _within_days(
        _read_jsonl(sorted(glob.glob(os.path.join(log_dir, "retrieval-*.jsonl")))), days)
    reports = _within_days(
        _read_jsonl(sorted(glob.glob(os.path.join(log_dir, "reports-*.jsonl")))), days)
    return retrieval, reports


# --- section builders (each returns a list of markdown lines) ------------------

def _feedback_clusters(reports: list[dict], top: int) -> list[str]:
    ok = [r for r in reports if r.get("event") == "report_feedback" and r.get("ok")]
    out = [f"## Feedback clusters ({len(ok)} reports)", ""]
    if not ok:
        out += ["_No recorded feedback._", ""]
        return out

    out.append("By signal type: " + ", ".join(
        f"`{k}` {v}" for k, v in Counter(_norm(r.get("signal_type")) for r in ok).most_common()))
    out.append("")
    out.append("By primary target: " + ", ".join(
        f"`{k}` {v}" for k, v in Counter(
            _norm((r.get("recommendation") or {}).get("primary_target")) for r in ok).most_common()))
    out.append("")
    outc = Counter(_norm(r.get("final_outcome")) for r in ok)
    unresolved = outc.get("not_fixed", 0) + outc.get("abandoned", 0) + outc.get("workaround", 0)
    out.append(f"Outcomes: " + ", ".join(f"`{k}` {v}" for k, v in outc.most_common())
               + f"  → **{unresolved}/{len(ok)} ({_pct(unresolved, len(ok))}) not cleanly fixed**")
    out += ["", "### Clusters (by dedup_fingerprint, largest first)", ""]

    clusters: dict[str, list[dict]] = defaultdict(list)
    for r in ok:
        clusters[r.get("dedup_fingerprint") or "(none)"].append(r)
    for fp, items in sorted(clusters.items(), key=lambda kv: -len(kv[1]))[:top]:
        sig = Counter(_norm(i.get("signal_type")) for i in items).most_common(1)[0][0]
        tgt = Counter(_norm((i.get("recommendation") or {}).get("primary_target"))
                      for i in items).most_common(1)[0][0]
        journeys = len({i.get("agent_journey_id") for i in items})
        sample = items[0]
        rec = (sample.get("recommendation") or {}).get("suggested_change", "")
        out.append(f"- **{fp}** ×{len(items)} ({journeys} journeys) — `{sig}` → `{tgt}`")
        out.append(f"  - problem: {sample.get('problem_summary', '')[:200]}")
        if rec:
            out.append(f"  - suggested: {rec[:200]}")
        exp = sample.get("expectation")
        if isinstance(exp, dict):
            out.append(f"  - expected: {str(exp.get('expected',''))[:120]} | actual: {str(exp.get('actual',''))[:120]}")
    out.append("")
    return out


def _retrieval_gaps(retrieval: list[dict], low_score: float, top: int) -> list[str]:
    calls = [r for r in retrieval if r.get("event") == "retrieve_docs" and r.get("ok")]
    out = [f"## Retrieval coverage gaps ({len(calls)} calls)", ""]
    if not calls:
        out += ["_No retrieval activity._", ""]
        return out

    empty = Counter()
    low = defaultdict(list)  # nquery -> [top_scores]
    freq = Counter()
    for c in calls:
        q = _norm(c.get("query"))
        if not q:
            continue
        freq[q] += 1
        ts = c.get("top_score")
        if c.get("n_results", 0) == 0 or ts is None:
            empty[q] += 1
        elif ts < low_score:
            low[q].append(ts)

    out.append(f"**Empty-result queries** (no chunks): {sum(empty.values())} calls")
    for q, n in empty.most_common(top):
        out.append(f"- ×{n} — {q[:160]}")
    out.append("")
    out.append(f"**Low-score queries** (top_score < {low_score}): {sum(len(v) for v in low.values())} calls")
    for q, scores in sorted(low.items(), key=lambda kv: -len(kv[1]))[:top]:
        avg = sum(scores) / len(scores)
        out.append(f"- ×{len(scores)} avg={avg:.3f} — {q[:160]}")
    out.append("")
    out.append("**Most frequent queries** (demand signal):")
    for q, n in freq.most_common(top):
        out.append(f"- ×{n} — {q[:160]}")
    out.append("")
    return out


def _operational(retrieval: list[dict], reports: list[dict]) -> list[str]:
    rejected = [r for r in reports if r.get("event") == "report_feedback" and not r.get("ok")]
    errors = [r for r in (retrieval + reports) if r.get("event") == "event_log_error"]
    out = ["## Operational health", ""]
    out.append(f"- Rejected feedback submissions: {len(rejected)}")
    for reason, n in Counter(r.get("reason", "?") for r in rejected).most_common():
        out.append(f"  - ×{n}: {reason}")
    out.append(f"- event_log_error lines (persistence failures): {len(errors)}")
    out.append("")
    return out


def build_digest(log_dir: str, days: int | None, low_score: float, top: int) -> str:
    retrieval, reports = load_events(log_dir, days)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    span = f"last {days}d" if days else "all time"
    head = [
        f"# MCP feedback triage digest",
        "",
        f"_Generated {now} · source `{log_dir}` · window {span}_",
        "",
        "A recommendation digest — not a decision. Cluster the signals, then edit "
        "How-to / Rules / Brief or file a bug. No issues are opened automatically.",
        "",
    ]
    body = (
        _feedback_clusters(reports, top)
        + _retrieval_gaps(retrieval, low_score, top)
        + _operational(retrieval, reports)
    )
    return "\n".join(head + body)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MCP feedback-loop triage digest (Phase C)")
    ap.add_argument("--log-dir", default=os.environ.get("LOG_DIR", ""),
                    help="Directory with retrieval-*.jsonl / reports-*.jsonl (default: $LOG_DIR)")
    ap.add_argument("--days", type=int, default=None, help="Only events newer than N days")
    ap.add_argument("--low-score", type=float, default=0.5, help="top_score below this is a gap")
    ap.add_argument("--top", type=int, default=15, help="Max rows per ranked list")
    ap.add_argument("--out", default=None, help="Write to file instead of stdout")
    args = ap.parse_args(argv)
    if not args.log_dir:
        ap.error("no --log-dir (or $LOG_DIR) given")

    digest = build_digest(args.log_dir, args.days, args.low_score, args.top)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(digest + "\n")
        print(f"wrote {args.out}")
    else:
        print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
