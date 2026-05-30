"""`lsfusion_report_feedback` — the feedback-loop sink (MCP-FEEDBACK-PLAN.md, Phase B).

When an agent accumulates errors during an lsFusion task (mostly failed `eval`
runs) and the user consents, it submits ONE self-contained, depersonalized
"journey": the eval errors it hit, the doc queries it tried, how it resolved
things, and a recommendation. A report is a RECOMMENDATION, not a decision — the
actual doc/code fix is decided by a human aggregating many reports + logs.

This tool lives on the central server (A) and is the single sink. The platform
app-server MCP and the IntelliJ plugin proxy it through (they hand-mirror the
schema, exactly like `retrieve_docs`).

Server side: assign a `report_id`, enforce anti-abuse caps, run a best-effort
redaction pass (defense-in-depth — NOT anonymization; the agent is instructed to
send no code/customer data), compute the canonical `dedup_fingerprint`, and
append the (redacted) report to the `reports` event stream. Consent + the
"when to report" trigger live in a workflow Rule served by `get_guidance`.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from settings import (
    FEEDBACK_ENABLED,
    REPORT_MAX_EVAL_ERRORS,
    REPORT_MAX_QUERIES,
    REPORT_CODE_EXCERPT_MAX_CHARS,
    REPORT_MAX_TOTAL_CHARS,
)
from tools.event_log import emit

TargetArtifact = Literal[
    "how-to", "rules", "brief", "paradigm", "language", "code-bug", "rag-retrieval",
    # Recommendation to improve the DIAGNOSTIC (message/location/hint), not the
    # validator itself — `eval` stays the validator/executor.
    "eval-error-message",
]

# Broad reinforcement-quality signal (single primary; nuance via expectation /
# secondary_targets / rationale). Not just doc gaps — also when the agent's
# mental model was wrong, or eval's error was unactionable.
SignalType = Literal[
    "doc-gap",
    "expectation-mismatch",
    "unclear-error",
    "missing-capability",
    "rag-retrieval",
    "other",
]


# Free-text / list caps (per-field, so a single oversized field is rejected at
# parse time with a clear error rather than only tripping the global byte cap).
_TXT = 4000   # long-ish free text (messages, suggestions, excerpts)
_SHORT = 2000  # short free text (summaries, queries, expectations)
_LIST = 30     # list lengths


class Expectation(BaseModel):
    """What the agent expected vs what lsFusion / a tool actually did."""
    expected: str = Field(..., max_length=_SHORT, description="What the agent expected (depersonalized).")
    actual: str = Field(..., max_length=_SHORT, description="What actually happened (depersonalized).")


class EvalError(BaseModel):
    """One error the agent hit while running lsFusion code via `eval`."""
    message: str = Field(..., max_length=_TXT, description="The error text (depersonalized).")
    phase: Literal["syntax", "semantic", "runtime", "unknown"] = Field(
        "unknown", description="Where it surfaced.")
    code_excerpt: Optional[str] = Field(
        None, max_length=_TXT, description="Tiny abstracted snippet if essential — NO full source / project code.")
    normalized_message: Optional[str] = Field(
        None, max_length=_SHORT, description="Optional normalized form (ids/literals stripped) for clustering.")


class RetrieveQuery(BaseModel):
    """A `retrieve_docs` query the agent ran while trying to fix the error."""
    query: str = Field(..., max_length=_SHORT, description="The query text.")
    returned_sources: Optional[List[str]] = Field(
        None, max_length=_LIST, description="Doc branches/files it surfaced, if noted.")
    usefulness: Optional[Literal["helpful", "irrelevant", "misleading", "incomplete"]] = Field(
        None, description="How useful the result was for this error.")


class Recommendation(BaseModel):
    """The agent's suggested fix — a HINT for triage clustering, not a decision."""
    primary_target: TargetArtifact = Field(..., description="Main artifact to change.")
    secondary_targets: List[TargetArtifact] = Field(
        default_factory=list, max_length=_LIST, description="Other plausibly-affected artifacts.")
    suggested_change: str = Field(..., max_length=_TXT, description="Concrete suggestion (depersonalized).")
    confidence: Literal["low", "medium", "high"] = Field("medium")
    rationale: Optional[str] = Field(None, max_length=_SHORT, description="Why, briefly.")


class ToolContext(BaseModel):
    """Typed (not a free dict) so arbitrary agent-supplied keys can't leak."""
    model_config = {"extra": "ignore"}  # drop unknown keys rather than store them
    eval_server_kind: Optional[str] = Field(None, max_length=100)
    eval_server_version: Optional[str] = Field(None, max_length=100)


class FeedbackReport(BaseModel):
    """A self-contained, depersonalized error/feedback report. NO code, file
    paths, schema/table/customer names, or secrets — only a doc-improvement
    signal plus public doc references."""
    agent_journey_id: str = Field(
        ..., max_length=200, description="Agent-generated id grouping this task's errors/queries.")
    signal_type: SignalType = Field(
        ..., description="What kind of reinforcement signal this is (routing hint, not the decision).")
    problem_summary: str = Field(..., max_length=_SHORT, description="Short depersonalized task description.")
    recommendation: Recommendation
    expectation: Optional[Expectation] = Field(
        None, description="For expectation-mismatch/unclear-error: expected vs actual.")
    eval_errors: List[EvalError] = Field(default_factory=list)
    retrieve_queries: List[RetrieveQuery] = Field(default_factory=list)
    retrieved_docs_summary: List[str] = Field(
        default_factory=list, max_length=_LIST,
        description="Short PUBLIC summaries of retrieved docs (no chunk bodies).")
    final_outcome: Literal["fixed", "not_fixed", "workaround", "abandoned", "unknown"] = Field(
        "unknown", description="How the task ended (guards against survivorship bias).")
    tool_context: Optional[ToolContext] = Field(None, description="Optional eval-server context.")
    client_dedup_hint: Optional[str] = Field(
        None, max_length=200, description="Optional agent hint; the SERVER computes the canonical dedup_fingerprint.")
    lsfusion_version: Optional[str] = Field(None, max_length=100)
    deployment_kind: Optional[str] = Field(None, max_length=100)
    agent: Optional[str] = Field(None, max_length=100, description="Reporting client name/version, e.g. claude-code.")
    n_eval_attempts: Optional[int] = None


class FeedbackOutput(BaseModel):
    report_id: str = Field(..., description="Server-assigned id for this submission.")
    status: Literal["recorded", "disabled", "rejected"] = Field(...)
    dedup_fingerprint: Optional[str] = Field(
        None, description="Server-computed clustering fingerprint (when recorded).")
    detail: Optional[str] = Field(None, description="Reason when disabled/rejected.")


# --- best-effort redaction (defense-in-depth, NOT anonymization) ---------------
# Keys whose values are server/enum/id-controlled and must NOT be mangled. NOTE:
# agent-supplied free text (client_dedup_hint, summaries, messages, …) is NOT here
# — it gets redacted. agent_journey_id is kept (opaque id by contract).
_KEEP_KEYS = {
    "agent_journey_id", "phase", "usefulness", "final_outcome",
    "primary_target", "secondary_targets", "confidence", "n_eval_attempts", "signal_type",
}
# Best-effort, NOT anonymization. Order: specific token shapes before the generic
# hex/key patterns. Known gaps are acceptable (the agent is instructed to send no
# secrets/code; this is a seatbelt).
_REDACTIONS = [
    (re.compile(r"(?i)\b(?:authorization|bearer)\s*[:=]?\s*[A-Za-z0-9._\-]{8,}"), "<redacted-auth>"),
    (re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"), "<redacted-jwt>"),
    (re.compile(r"\b(?:gh[posru]|github_pat)_[A-Za-z0-9_]{20,}\b"), "<redacted-token>"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "<redacted-aws-key>"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), "<redacted-slack>"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "<redacted-email>"),
    (re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{12,}\b"), "<redacted-key>"),
    (re.compile(r"(?i)\b(token|secret|api[_-]?key|password|passwd|pwd)\b\s*[:=]\s*\S+"), r"\1=<redacted>"),
    (re.compile(r"\b\w+://[^\s/@]+:[^\s/@]+@\S+"), "<redacted-uri-creds>"),
    (re.compile(r"\b[A-Fa-f0-9]{32,}\b"), "<redacted-hex>"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "<redacted-ip>"),
    (re.compile(r"(?:/home/|/Users/|/var/|/srv/|/opt/|[A-Za-z]:\\)[^\s:,)\]\"']+"), "<redacted-path>"),
    (re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"),
     "<redacted-uuid>"),
]


def _redact_text(s: str) -> str:
    for pat, repl in _REDACTIONS:
        s = pat.sub(repl, s)
    return s


def _redact(o):
    if isinstance(o, str):
        return _redact_text(o)
    if isinstance(o, list):
        return [_redact(x) for x in o]
    if isinstance(o, dict):
        # Redact both keys and values, except server/enum/id-controlled keys.
        return {
            (k if k in _KEEP_KEYS else _redact_text(k)): (v if k in _KEEP_KEYS else _redact(v))
            for k, v in o.items()
        }
    return o


def _norm(s: Optional[str]) -> str:
    return " ".join((s or "").lower().split())


def _dedup_fingerprint(red: dict) -> str:
    errs = red.get("eval_errors") or []
    first_err = ""
    if errs:
        # Prefer the normalized form (ids/literals stripped) over the raw message.
        first_err = _norm(errs[0].get("normalized_message") or errs[0].get("message"))
    target = (red.get("recommendation") or {}).get("primary_target", "")
    exp = red.get("expectation") or {}
    # Cluster on the stable, discriminating fields. problem_summary is deliberately
    # EXCLUDED — agent prose is inconsistent and would over-fragment clusters.
    basis = "|".join([
        red.get("signal_type", ""),
        first_err,
        target,
        _norm(exp.get("expected")),
        _norm(exp.get("actual")),
    ])
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def _byte_len(s: str) -> int:
    return len(s.encode("utf-8"))


def _reject_reason(report: FeedbackReport) -> Optional[str]:
    if len(report.eval_errors) > REPORT_MAX_EVAL_ERRORS:
        return f"too many eval_errors (>{REPORT_MAX_EVAL_ERRORS})"
    if len(report.retrieve_queries) > REPORT_MAX_QUERIES:
        return f"too many retrieve_queries (>{REPORT_MAX_QUERIES})"
    for e in report.eval_errors:
        if e.code_excerpt and len(e.code_excerpt) > REPORT_CODE_EXCERPT_MAX_CHARS:
            return f"code_excerpt exceeds {REPORT_CODE_EXCERPT_MAX_CHARS} chars (no source dumps)"
    # Size cap in BYTES (multibyte chars must not slip past a char-count cap).
    if _byte_len(report.model_dump_json()) > REPORT_MAX_TOTAL_CHARS:
        return f"payload exceeds {REPORT_MAX_TOTAL_CHARS} bytes"
    return None


def _rejected(report_id: str, reason: str, payload_size: int = 0) -> FeedbackOutput:
    # Minimal rejected event — NO user payload (see codex review r8).
    emit("report_feedback",
         {"report_id": report_id, "reason": reason, "payload_size": payload_size},
         stream="reports", ok=False)
    return FeedbackOutput(report_id=report_id, status="rejected", detail=reason)


def report_feedback_tool(report: FeedbackReport) -> FeedbackOutput:
    """Validate, redact, fingerprint and record one feedback report. See module docstring."""
    if not FEEDBACK_ENABLED:
        return FeedbackOutput(
            report_id="", status="disabled",
            detail="report_feedback is disabled on this server")

    report_id = "rpt_" + uuid.uuid4().hex[:16]
    try:
        reason = _reject_reason(report)
        if reason:
            return _rejected(report_id, reason, _byte_len(report.model_dump_json()))

        red = _redact(report.model_dump(exclude_none=True))
        # Backstop: redaction can grow fields (placeholders) — bound the STORED record too.
        stored = json.dumps(red, ensure_ascii=False)
        if _byte_len(stored) > REPORT_MAX_TOTAL_CHARS:
            return _rejected(report_id, f"redacted payload exceeds {REPORT_MAX_TOTAL_CHARS} bytes",
                             _byte_len(stored))

        dedup = _dedup_fingerprint(red)
        red["report_id"] = report_id
        red["dedup_fingerprint"] = dedup
        emit("report_feedback", red, stream="reports", ok=True)
        return FeedbackOutput(report_id=report_id, status="recorded", dedup_fingerprint=dedup)
    except Exception:  # noqa: BLE001 — never 500 the tool on processing failure
        # Don't emit here: the failure may BE in emit; re-calling it could re-raise.
        return FeedbackOutput(report_id=report_id, status="rejected", detail="processing error")
