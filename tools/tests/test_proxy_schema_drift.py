"""Drift guard: the hand-mirrored `lsfusion_report_feedback` schema in the two
proxy layers (platform `MCPDispatcher.java`, plugin `McpBaseService.java` +
`McpToolset.kt`) must stay in sync with the central source-of-truth schema
(mcp/tools/feedback.py).

The central Pydantic models are authoritative. For each proxy we slice the
region that declares the tool, tokenize it, and assert every central field name
and every central enum value APPEARS there (a coverage/subset check — it catches
"central added/renamed a field or enum value, proxy forgot it", the drift that
actually bites). Reverse drift (proxy keeps a field central dropped) is not
checked here.

Runs only in the aggregate super-workspace where the sibling repos exist; a
standalone mcp checkout skips it.
"""
from __future__ import annotations

import pathlib
import re

import pytest

from tools.feedback import FeedbackOutput, FeedbackReport

# tools/tests/ -> tools -> mcp -> <aggregate>
_AGG = pathlib.Path(__file__).resolve().parents[3]
PLATFORM = _AGG / "platform/server/src/main/java/lsfusion/server/physics/admin/mcp/MCPDispatcher.java"
PLUGIN_JAVA = _AGG / "plugin-idea/src/com/lsfusion/mcp/McpBaseService.java"
PLUGIN_KT = _AGG / "plugin-idea/src/com/lsfusion/mcp/McpToolset.kt"

_IDENT = re.compile(r"[a-z][a-z0-9_-]*")


def _enum_values(prop: dict) -> set[str]:
    out: set[str] = set(prop.get("enum") or [])
    for sub in prop.get("anyOf") or []:
        out |= _enum_values(sub)
    items = prop.get("items")
    if isinstance(items, dict):
        out |= _enum_values(items)
    return out


def _contract(model) -> tuple[set[str], set[str]]:
    """(field names, enum values) across a model and its nested $defs."""
    schema = model.model_json_schema()
    fields: set[str] = set()
    enums: set[str] = set()
    for m in [schema, *(schema.get("$defs") or {}).values()]:
        for name, prop in (m.get("properties") or {}).items():
            fields.add(name)
            enums |= _enum_values(prop)
    return fields, enums


def _brace_slice(text: str, marker: str) -> str:
    """Body from `marker` to its brace-matched close (a method)."""
    start = text.index(marker)
    depth = 0
    i = text.index("{", start)
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start:j + 1]
    return text[start:]


def _tokens(region: str) -> set[str]:
    return set(_IDENT.findall(region))


def _assert_covers(region_tokens: set[str], fields: set[str], enums: set[str], who: str):
    missing_f = sorted(fields - region_tokens)
    missing_e = sorted(enums - region_tokens)
    assert not missing_f, f"{who}: missing central fields {missing_f}"
    assert not missing_e, f"{who}: missing central enum values {missing_e}"


def _need(path: pathlib.Path):
    if not path.exists():
        pytest.skip(f"sibling repo file not present: {path}")
    return path.read_text(encoding="utf-8")


IN_FIELDS, IN_ENUMS = _contract(FeedbackReport)
OUT_FIELDS, OUT_ENUMS = _contract(FeedbackOutput)


def test_platform_descriptor_covers_central_input():
    text = _need(PLATFORM)
    region = _brace_slice(text, "JSONObject reportFeedbackDescriptor(")  # definition, not the call site
    _assert_covers(_tokens(region), IN_FIELDS, IN_ENUMS, "platform MCPDispatcher")


def test_plugin_java_descriptor_covers_central_input_and_output():
    text = _need(PLUGIN_JAVA)
    region = _brace_slice(text, "JSONObject buildReportFeedbackToolDescriptor(")  # definition, not the call site
    _assert_covers(_tokens(region), IN_FIELDS | OUT_FIELDS, IN_ENUMS | OUT_ENUMS, "plugin McpBaseService")


def test_plugin_kotlin_dtos_cover_central_input_and_output():
    text = _need(PLUGIN_KT)
    # The Fb* DTO block sits between RetrieveDocsOutput and `class McpToolset`.
    region = text[text.index("data class FbExpectation"):text.index("class McpToolset")]
    _assert_covers(_tokens(region), IN_FIELDS | OUT_FIELDS, IN_ENUMS | OUT_ENUMS, "plugin McpToolset")


def test_central_contract_is_nonempty_guard():
    # Guard: if the central models lose their fields/enums, the subset checks above
    # would vacuously pass — assert the contract itself is substantial.
    assert {"signal_type", "problem_summary", "recommendation", "agent_journey_id"} <= IN_FIELDS
    assert {"doc-gap", "eval-error-message", "expectation-mismatch"} <= IN_ENUMS
    assert {"report_id", "status"} <= OUT_FIELDS
