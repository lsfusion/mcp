"""Drift guard: the hand-mirrored `lsfusion_report_feedback` and
`lsfusion_retrieve_docs` schemas in the two proxy layers (platform
`MCPDispatcher.java`, plugin `McpBaseService.java` + `McpToolset.kt`) must stay
in sync with the central source-of-truth schemas (mcp/tools/feedback.py,
mcp/tools/rag_retrieve.py).

The central Pydantic models are authoritative. For each proxy we slice the
region that declares the tool, tokenize it, and assert every central field name
and every central enum value APPEARS there (a coverage/subset check — it catches
"central added/renamed a field or enum value, proxy forgot it", the drift that
actually bites). Reverse drift (proxy keeps a field central dropped) is not
checked here.

Runs only in the aggregate super-workspace where the sibling repos exist; a
standalone mcp checkout skips it. The sibling roots can be redirected with
`LSF_PLATFORM_ROOT` / `LSF_PLUGIN_ROOT`, so the guard can be pointed at the
worktree that actually carries a mirroring change instead of whatever branch
the workspace checkout happens to be on.
"""
from __future__ import annotations

import inspect
import os
import pathlib
import re

import pytest

from tools.feedback import FeedbackOutput, FeedbackReport
from tools.guidance import BRANCH_PREFIX
from tools.rag_retrieve import RetrieveDocsOutput, retrieve_docs_tool

# tools/tests/ -> tools -> mcp -> <aggregate>
_AGG = pathlib.Path(__file__).resolve().parents[3]
_ROOTS_OVERRIDDEN = bool(os.environ.get("LSF_PLATFORM_ROOT") or os.environ.get("LSF_PLUGIN_ROOT"))
_PLATFORM_ROOT = pathlib.Path(os.environ.get("LSF_PLATFORM_ROOT", _AGG / "platform"))
_PLUGIN_ROOT = pathlib.Path(os.environ.get("LSF_PLUGIN_ROOT", _AGG / "plugin-idea"))
PLATFORM = _PLATFORM_ROOT / "server/src/main/java/lsfusion/server/physics/admin/mcp/MCPDispatcher.java"
PLUGIN_JAVA = _PLUGIN_ROOT / "src/com/lsfusion/mcp/McpBaseService.java"
PLUGIN_KT = _PLUGIN_ROOT / "src/com/lsfusion/mcp/McpToolset.kt"

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


def _assert_declares(region: str, keys: set[str], who: str, *, kotlin: bool = False):
    """Token coverage alone is too weak: a name mentioned only in a comment or a
    description string counts. Require each key to appear where a key is
    actually declared — `.put("<key>"` in the JSONObject builders, `val <key>`
    in the Kotlin DTO."""
    missing = sorted(k for k in keys
                     if (f"val {k}" if kotlin else f'.put("{k}"') not in region)
    assert not missing, f"{who}: names present but never declared as fields: {missing}"


def _assert_covers(region_tokens: set[str], fields: set[str], enums: set[str], who: str):
    missing_f = sorted(fields - region_tokens)
    missing_e = sorted(enums - region_tokens)
    assert not missing_f, f"{who}: missing central fields {missing_f}"
    assert not missing_e, f"{who}: missing central enum values {missing_e}"


def _need(path: pathlib.Path):
    if not path.exists():
        # An explicitly pointed root that does not exist is a typo, not an
        # absent sibling: skipping it would turn the whole guard green.
        if _ROOTS_OVERRIDDEN:
            raise AssertionError(f"root was set explicitly but file is missing: {path}")
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


# --- retrieve_docs ---------------------------------------------------------
#
# The input side has no Pydantic model — the signature IS the contract, so read
# it, rather than restating it here where it would silently stop tracking.
# `id` and `exclude_ids` are the ones that matter:
# a proxy that forgets `exclude_ids` cannot pass it at all (both descriptors set
# `additionalProperties: false`), and one that forgets `id` drops it silently on
# the way back, which is worse than failing.
RETRIEVE_IN_FIELDS = set(inspect.signature(retrieve_docs_tool).parameters)
RETRIEVE_OUT_FIELDS, RETRIEVE_OUT_ENUMS = _contract(RetrieveDocsOutput)


def test_platform_retrieve_descriptor_covers_central_input():
    text = _need(PLATFORM)
    region = _brace_slice(text, "JSONObject retrieveDocsDescriptor(")
    who = "platform MCPDispatcher (retrieve_docs)"
    _assert_covers(_tokens(region), RETRIEVE_IN_FIELDS, set(), who)
    _assert_declares(region, RETRIEVE_IN_FIELDS, who)


def test_plugin_java_retrieve_descriptor_covers_central_input_and_output():
    text = _need(PLUGIN_JAVA)
    region = _brace_slice(text, "JSONObject buildRetrieveDocsToolDescriptor(")
    who = "plugin McpBaseService (retrieve_docs)"
    _assert_covers(_tokens(region), RETRIEVE_IN_FIELDS | RETRIEVE_OUT_FIELDS,
                   RETRIEVE_OUT_ENUMS, who)
    _assert_declares(region, RETRIEVE_IN_FIELDS | RETRIEVE_OUT_FIELDS, who)


def test_plugin_kotlin_retrieve_dtos_cover_central_output():
    text = _need(PLUGIN_KT)
    region = text[text.index("data class RemoteDocItem"):text.index("// report_feedback DTOs")]
    who = "plugin McpToolset (retrieve_docs DTOs)"
    _assert_covers(_tokens(region), RETRIEVE_OUT_FIELDS, RETRIEVE_OUT_ENUMS, who)
    _assert_declares(region, RETRIEVE_OUT_FIELDS, who, kotlin=True)


def test_plugin_kotlin_wrapper_forwards_the_exclusion_list():
    # The DTO block above cannot see the call site: the Kotlin wrapper has to
    # both accept the parameter and put it on the wire under its snake_case name.
    text = _need(PLUGIN_KT)
    region = _brace_slice(text, "suspend fun retrieveDocs(")
    assert "excludeIds" in region, "plugin McpToolset: retrieveDocs takes no exclusion list"
    assert '"exclude_ids"' in region, "plugin McpToolset: exclusion list never reaches the wire"


# --- get_guidance ----------------------------------------------------------
#
# `get_guidance` had no guard here at all, because for its whole life it took no
# parameters and there was nothing to drift. Now it takes two, and they must be
# DECLARED in every proxy even though all of them forward args verbatim: each
# descriptor sets `additionalProperties: false`, so a parameter a proxy does not
# declare is one a strict client cannot pass. A proxy that misses them silently
# offers only the zero-argument call, and the per-area articles — the whole
# point of the redesign — stay unreachable through it.
GUIDANCE_PARAMS = set(BRANCH_PREFIX)


def test_central_guidance_contract_is_the_two_branches():
    # Same vacuity guard as above: if the branches vanish, the checks below pass
    # for the wrong reason.
    assert GUIDANCE_PARAMS == {"rules", "brief"}


def test_platform_guidance_descriptor_declares_both_branches():
    text = _need(PLATFORM)
    region = _brace_slice(text, "JSONObject getGuidanceDescriptor(")
    _assert_declares(region, GUIDANCE_PARAMS, "platform MCPDispatcher (get_guidance)")


def test_plugin_java_guidance_descriptor_declares_both_branches():
    text = _need(PLUGIN_JAVA)
    region = _brace_slice(text, "JSONObject buildGetGuidanceToolDescriptor(")
    _assert_declares(region, GUIDANCE_PARAMS, "plugin McpBaseService (get_guidance)")


def test_plugin_kotlin_wrapper_forwards_the_article_name():
    # The Kotlin path declares its schema through annotations on the function,
    # so the parameters and the wire names both live at the call site.
    text = _need(PLUGIN_KT)
    region = _brace_slice(text, "suspend fun getGuidance(")
    missing = sorted(p for p in GUIDANCE_PARAMS if p not in region)
    assert not missing, f"plugin McpToolset: getGuidance cannot name an article: {missing}"
