"""Drift guard: the hand-mirrored `lsfusion_report_feedback` and
`lsfusion_retrieve_docs` schemas in the two proxy layers (platform
`MCPDispatcher.java`, plugin `McpBaseService.java` + `McpToolset.kt`) must stay
in sync with the central source-of-truth schemas (mcp/tools/feedback.py,
mcp/tools/rag_retrieve.py).

The central Pydantic models are authoritative. For each proxy we slice the
region that declares the tool, tokenize it, and assert every central field name
and every central enum value APPEARS there (a coverage/subset check — it catches
"central added/renamed a field or enum value, proxy forgot it", the drift that
actually bites). Reverse drift (proxy keeps a field central dropped) is
checked only where it breaks a call: see "reverse drift" and "what a released
plugin depends on" below.

The checks that read the proxies carry the `siblings` mark and find the sibling
checkouts through tools/tests/siblings.py: next to mcp by default, as in the
super-workspace, or wherever `LSF_PLATFORM_ROOT` / `LSF_PLUGIN_ROOT` point, so
the guard can be aimed at the worktree that actually carries a mirroring change
instead of whatever branch the workspace checkout happens to be on. A
standalone mcp checkout skips them; with a root set explicitly, a missing file
fails instead. CI checks out both siblings' master and sets both roots.

Those also marked `released` are what a plugin already released depends on;
CI runs them against the plugin's latest release tag as well (see "what a
released plugin depends on" at the end).
"""
from __future__ import annotations

import inspect
import pathlib
import re

import pytest

from tools.feedback import FeedbackOutput, FeedbackReport
from tools.guidance import BRANCH_PREFIX
from tools.rag_retrieve import RetrieveDocsOutput, retrieve_docs_tool
from tools.tests.siblings import PLATFORM_ROOT, PLUGIN_ROOT, need

siblings = pytest.mark.siblings
# Also run against the latest RELEASED plugin (see "what a released plugin
# depends on" below): a failure there breaks a plugin users already have.
released = pytest.mark.released

PLATFORM = PLATFORM_ROOT / "server/src/main/java/lsfusion/server/physics/admin/mcp/MCPDispatcher.java"
PLUGIN_JAVA = PLUGIN_ROOT / "src/com/lsfusion/mcp/McpBaseService.java"
PLUGIN_KT = PLUGIN_ROOT / "src/com/lsfusion/mcp/McpToolset.kt"

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


def _skip_literal(text: str, i: int) -> int:
    """Index past the string or char literal, or the comment, that starts at `i`;
    `i` itself when none does. Kotlin raw strings included; a string template
    with a string inside it is not."""
    if text.startswith('"""', i):
        return text.index('"""', i + 3) + 3
    if text.startswith("//", i):
        end = text.find("\n", i)
        return len(text) if end < 0 else end
    if text.startswith("/*", i):
        return text.index("*/", i + 2) + 2
    if text[i] in "\"'":
        j = i + 1
        while text[j] != text[i]:
            j += 2 if text[j] == "\\" else 1
        return j + 1
    return i


def _closing(text: str, i: int, opening: str, closing: str) -> int | None:
    """Index of the `closing` bracket that matches an `opening` one just before
    `i`, past literals and comments; None when it never closes."""
    depth = 1
    while i < len(text):
        j = _skip_literal(text, i)
        if j != i:
            i = j
            continue
        if text[i] == opening:
            depth += 1
        elif text[i] == closing:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def _brace_slice(text: str, marker: str) -> str:
    """Body from `marker` to its brace-matched close (a method)."""
    start = i = text.index(marker)
    while text[i] != "{":  # past literals too: a parameter's description may hold one
        i = max(_skip_literal(text, i), i + 1)
    end = _closing(text, i + 1, "{", "}")
    return text[start:] if end is None else text[start:end + 1]


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


def _need(path: pathlib.Path) -> str:
    return need(path).read_text(encoding="utf-8")


IN_FIELDS, IN_ENUMS = _contract(FeedbackReport)
OUT_FIELDS, OUT_ENUMS = _contract(FeedbackOutput)


@siblings
def test_platform_descriptor_covers_central_input():
    text = _need(PLATFORM)
    region = _brace_slice(text, "JSONObject reportFeedbackDescriptor(")  # definition, not the call site
    _assert_covers(_tokens(region), IN_FIELDS, IN_ENUMS, "platform MCPDispatcher")


@siblings
def test_plugin_java_descriptor_covers_central_input_and_output():
    text = _need(PLUGIN_JAVA)
    region = _brace_slice(text, "JSONObject buildReportFeedbackToolDescriptor(")  # definition, not the call site
    _assert_covers(_tokens(region), IN_FIELDS | OUT_FIELDS, IN_ENUMS | OUT_ENUMS, "plugin McpBaseService")


@siblings
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


@siblings
def test_platform_retrieve_descriptor_covers_central_input():
    text = _need(PLATFORM)
    region = _brace_slice(text, "JSONObject retrieveDocsDescriptor(")
    who = "platform MCPDispatcher (retrieve_docs)"
    _assert_covers(_tokens(region), RETRIEVE_IN_FIELDS, set(), who)
    _assert_declares(region, RETRIEVE_IN_FIELDS, who)


@siblings
def test_plugin_java_retrieve_descriptor_covers_central_input_and_output():
    text = _need(PLUGIN_JAVA)
    region = _brace_slice(text, "JSONObject buildRetrieveDocsToolDescriptor(")
    who = "plugin McpBaseService (retrieve_docs)"
    _assert_covers(_tokens(region), RETRIEVE_IN_FIELDS | RETRIEVE_OUT_FIELDS,
                   RETRIEVE_OUT_ENUMS, who)
    _assert_declares(region, RETRIEVE_IN_FIELDS | RETRIEVE_OUT_FIELDS, who)


@siblings
def test_plugin_kotlin_retrieve_dtos_cover_central_output():
    text = _need(PLUGIN_KT)
    region = text[text.index("data class RemoteDocItem"):text.index("// report_feedback DTOs")]
    who = "plugin McpToolset (retrieve_docs DTOs)"
    _assert_covers(_tokens(region), RETRIEVE_OUT_FIELDS, RETRIEVE_OUT_ENUMS, who)
    _assert_declares(region, RETRIEVE_OUT_FIELDS, who, kotlin=True)


@siblings
def test_plugin_kotlin_wrapper_takes_every_parameter_by_its_central_name():
    # The DTO block above cannot see the call site. The IDE's MCP server names
    # an argument after its Kotlin parameter and drops any key that matches
    # none, without a word: the wrapper took the list as `excludeIds`, every
    # `exclude_ids` an agent sent was lost, and paging by `status` returned the
    # same chunks again. So each central parameter must be a Kotlin parameter of
    # the same name (`queries` and `articles` are extra list halves of `query`
    # and `article`), and reach the wire under that name.
    text = _need(PLUGIN_KT)
    params = _call_args(text, "suspend fun retrieveDocs(")
    assert params is not None, "plugin McpToolset: no retrieveDocs"
    who = "plugin McpToolset (retrieveDocs)"
    missing = sorted(RETRIEVE_IN_FIELDS - set(re.findall(r"(\w+)\s*:", _code(params, literals=False))))
    assert not missing, f"{who}: no Kotlin parameter named as central's {missing}"
    _assert_declares(_brace_slice(text, "suspend fun retrieveDocs("), RETRIEVE_IN_FIELDS, who)


# --- get_guidance ----------------------------------------------------------
#
# `get_guidance` had no guard here at all, because for its whole life it took no
# parameters and there was nothing to drift. Now it takes two, and they must be
# DECLARED in every proxy even though all of them forward args verbatim: each
# descriptor sets `additionalProperties: false`, so a parameter a proxy does not
# declare is one a strict client cannot pass. A proxy that misses them silently
# offers only the zero-argument call, and the per-area articles — the whole
# point of the redesign — stay unreachable through it.
# Read from the tool's OWN signature rather than restated here. The earlier
# version derived this from `BRANCH_PREFIX`, which made it a guard over the two
# guidance branches specifically — so when `get_guidance` grew a third
# parameter the guard had nothing to say about it, and a proxy could ship
# without it exactly as the proxies once shipped without `rules` and `brief`.
# Three separate field reports came from that hole: the article instructed a
# call the installed tool could not accept. Deriving the set means the next
# parameter is covered on the day it is added.
def _guidance_params() -> set[str]:
    import server  # imported lazily: the module builds the MCP app at import
    fn = getattr(server.lsfusion_get_guidance, "fn", server.lsfusion_get_guidance)
    return set(inspect.signature(fn).parameters)


GUIDANCE_PARAMS = _guidance_params()


def test_central_guidance_contract_covers_every_named_read():
    # Vacuity guard: if the parameters vanish, the checks below pass for the
    # wrong reason. Reference articles are NOT among them — naming one is
    # `retrieve_docs(article=...)`, because a reference article is not sized to
    # arrive whole and is delivered as its chunks, paged.
    assert GUIDANCE_PARAMS
    assert set(BRANCH_PREFIX) <= GUIDANCE_PARAMS


@siblings
def test_platform_guidance_descriptor_declares_every_parameter():
    text = _need(PLATFORM)
    region = _brace_slice(text, "JSONObject getGuidanceDescriptor(")
    _assert_declares(region, GUIDANCE_PARAMS, "platform MCPDispatcher (get_guidance)")


@siblings
def test_plugin_java_guidance_descriptor_declares_every_parameter():
    text = _need(PLUGIN_JAVA)
    region = _brace_slice(text, "JSONObject buildGetGuidanceToolDescriptor(")
    _assert_declares(region, GUIDANCE_PARAMS, "plugin McpBaseService (get_guidance)")


@siblings
def test_plugin_kotlin_wrapper_forwards_the_article_name():
    # The Kotlin path declares its schema through annotations on the function,
    # so the parameters and the wire names both live at the call site.
    text = _need(PLUGIN_KT)
    region = _brace_slice(text, "suspend fun getGuidance(")
    missing = sorted(p for p in GUIDANCE_PARAMS if p not in region)
    assert not missing, f"plugin McpToolset: getGuidance cannot name an article: {missing}"


# --- reverse drift ---------------------------------------------------------
#
# Everything above is a SUBSET check: it catches "central added a field, the
# proxy forgot it". The opposite direction — the proxy still advertising
# something central dropped — is invisible to it, and this migration is exactly
# that shape: `brief` and `rules` left the search corpus, and a proxy that keeps
# offering them as `type` values invites a call the central server now rejects.
# The value is checked where an enum value actually lives (`.put("<v>")` inside
# the JSONArray), not as a bare token: both descriptions legitimately mention
# the two branches in prose, to say where they went.
@siblings
def test_platform_retrieve_enum_no_longer_offers_the_guidance_branches():
    region = _brace_slice(_need(PLATFORM), "JSONObject retrieveDocsDescriptor(")
    still = [v for v in ("brief", "rules") if f'.put("{v}")' in region]
    assert not still, f"platform MCPDispatcher: `type` still offers {still}"


@siblings
def test_plugin_java_retrieve_enum_no_longer_offers_the_guidance_branches():
    region = _brace_slice(_need(PLUGIN_JAVA), "JSONObject buildRetrieveDocsToolDescriptor(")
    still = [v for v in ("brief", "rules") if f'.put("{v}")' in region]
    assert not still, f"plugin McpBaseService: `type` still offers {still}"


def test_central_no_longer_searches_the_guidance_branches():
    # Vacuity guard for the two above.
    from tools.rag_retrieve import ALLOWED_TYPES
    assert not {"brief", "rules"} & set(ALLOWED_TYPES)


# --- the docs say the same thing as the tools ------------------------------
#
# Two ways the published guidance can quietly contradict the server, both hit
# during this migration. The instructions kept telling the assistant to make a
# call that now raises — rule B.2 of the top rules article, the literal
# documentation-lookup procedure, still said
# `lsfusion_retrieve_docs(type='rules', query='<area>')`. And the map is
# hand-maintained beside the article files, so an article nothing points at is
# invisible (which is what started all of this) and a mapped article that does
# not exist is a guaranteed miss on a name the assistant read from the map.
_GUIDANCE_DOCS = PLATFORM_ROOT / "docs"


def _guidance_articles(branch: str) -> set[str]:
    d = need(_GUIDANCE_DOCS / branch / "en")
    prefix = branch.capitalize() + "_"
    return {p.stem[len(prefix):] for p in d.glob(f"{prefix}*.md")}


def _mapped_names(branch: str) -> set[str]:
    top = _need(_GUIDANCE_DOCS / branch / "en" / f"{branch.capitalize()}.md")
    # First column of the map table: | `name` | ...
    return set(re.findall(r"^\|\s*`([a-z0-9_]+)`\s*\|", top, re.M))


@siblings
@pytest.mark.parametrize("branch", ["rules", "brief"])
def test_the_map_lists_exactly_the_articles_that_exist(branch):
    mapped, actual = _mapped_names(branch), _guidance_articles(branch)
    assert mapped == actual, (
        f"{branch}: map lists {sorted(mapped)}, folder has {sorted(actual)} — "
        f"an unmapped article is invisible, a mapped one that is missing is a dead name")


@siblings
@pytest.mark.parametrize("branch", ["rules", "brief"])
def test_no_guidance_article_still_points_at_the_retired_lookup(branch):
    d = need(_GUIDANCE_DOCS / branch)
    stale = [str(p.relative_to(_GUIDANCE_DOCS)) for p in d.rglob("*.md")
             if "type='rules'" in p.read_text(encoding="utf-8")
             or "type='brief'" in p.read_text(encoding="utf-8")]
    assert not stale, f"these still instruct a call the server rejects: {stale}"



# --- the batch cap is a NUMBER, and numbers drift too -------------------------
#
# The proxies cannot read BATCH_MAX_QUERIES, so they hard-code it. Field
# coverage cannot see a wrong number; this can. Note the constant is also
# settable from the environment — the guard compares against whatever value the
# central server would actually enforce in this run.
from settings import BATCH_MAX_QUERIES


@siblings
def test_platform_query_max_items_matches_the_central_cap():
    region = _brace_slice(_need(PLATFORM), "JSONObject retrieveDocsDescriptor(")
    m = re.search(r'\.put\("maxItems",\s*(\d+)\)', region)
    assert m, "platform MCPDispatcher: `query` array branch declares no maxItems"
    assert int(m.group(1)) == BATCH_MAX_QUERIES


@siblings
def test_plugin_java_query_max_items_matches_the_central_cap():
    region = _brace_slice(_need(PLUGIN_JAVA), "JSONObject buildRetrieveDocsToolDescriptor(")
    m = re.search(r'\.put\("maxItems",\s*(\d+)\)', region)
    assert m, "plugin McpBaseService: `query` array branch declares no maxItems"
    assert int(m.group(1)) == BATCH_MAX_QUERIES


@siblings
def test_plugin_kotlin_states_the_cap_in_prose():
    # The annotation path has no machine constraint on the list, so the
    # number lives only in the description there — check it is the right one.
    region = _brace_slice(_need(PLUGIN_KT), "suspend fun retrieveDocs(")
    assert f"At most {BATCH_MAX_QUERIES} DISTINCT" in region


# The checks above compare NAMES, so a proxy can carry a tool description from
# an older contract and stay green. That is what happened to `get_guidance`: the
# central text kept advertising an `article` selector that had moved to
# `retrieve_docs`, the proxies copied it, and agents passed `article` to a tool
# that silently ignored it. Descriptions are part of the contract a model
# reads, so the proxies mirror them verbatim.
def _central_description(tool: str) -> str:
    import asyncio
    import importlib
    tools = asyncio.run(importlib.import_module("server").mcp.list_tools())
    return {t.name: t for t in tools}[tool].description


def _literal_starting(text: str, opening: str) -> str:
    m = re.search(r'"(' + re.escape(opening) + r'(?:[^"\\]|\\.)*)"', text)
    assert m, f"no string literal starting {opening!r}"
    body = m.group(1)
    # Only the two escapes the descriptions need. Anything else (`\n`, `\t`, a
    # stray backslash) changes the text the JVM serves, so it must not be
    # decoded away into a match.
    stray = set(re.findall(r'\\(.)', body)) - {'"', '\\'}
    assert not stray, f"unexpected escapes in the literal: {sorted(stray)}"
    return re.sub(r'\\(.)', r'\1', body)


def _tools() -> dict:
    import asyncio
    import importlib
    return {t.name: t for t in asyncio.run(importlib.import_module("server").mcp.list_tools())}


# The Kotlin toolset cannot type a parameter as "string or list", so it splits
# `query` into `query`/`queries` and `article` into `article`/`articles`. Its
# retrieve_docs description names both forms; everything else is the central
# text. Each central fragment is asserted present, so rewriting it centrally
# fails here instead of leaving the Kotlin wording behind.
_KOTLIN_RETRIEVE = [
    ("`query` alone searches", "`query` (or `queries`) alone searches"),
    ("`article` alone WALKS the named articles from the top in document order — omit `query` to traverse;",
     "`article` (or `articles`) alone WALKS the named articles from the top in document order — omit both `query` and `queries` to traverse;"),
    ("Keeping `query` while adding `article` gets you a ranked search",
     "Keeping a query while adding an article gets you a ranked search"),
]


def _expected(tool: str, path) -> str:
    text = _tools()[tool].description
    if tool == "lsfusion_retrieve_docs" and path == PLUGIN_KT:
        for central, kotlin in _KOTLIN_RETRIEVE:
            assert central in text, f"central wording moved, revisit the Kotlin adaptation: {central!r}"
            text = text.replace(central, kotlin)
    return text


_DESCRIBED = [("lsfusion_get_guidance", "Read ONE lsFusion"),
              ("lsfusion_retrieve_docs", "Search official lsFusion documentation")]


@siblings
@pytest.mark.parametrize("path", [PLATFORM, PLUGIN_JAVA, PLUGIN_KT], ids=["platform", "plugin-java", "plugin-kotlin"])
@pytest.mark.parametrize("tool,opening", _DESCRIBED, ids=[t for t, _ in _DESCRIBED])
def test_proxy_mirrors_the_central_tool_description(path, tool, opening):
    assert _literal_starting(_need(path), opening) == _expected(tool, path)


def test_tool_descriptions_name_foreign_parameters_only_with_their_tool():
    # A parameter of ANOTHER tool may be mentioned only in a sentence that also
    # names that tool — otherwise a reader takes it for one of this tool's own,
    # which is exactly how `article` got passed to get_guidance.
    tools = _tools()
    for name, tool in tools.items():
        own = set(tool.inputSchema["properties"])
        for sentence in re.split(r"(?<=\.)\s+", tool.description):
            for p in set(re.findall(r"`([a-z_]+)`", sentence)) - own:
                owners = [o for o, t in tools.items() if o != name and p in t.inputSchema["properties"]]
                if owners:
                    assert any(o in sentence for o in owners), \
                        f"{name}: `{p}` belongs to {owners} but the sentence does not say so: {sentence!r}"



def test_tool_descriptions_call_tools_only_with_their_own_parameters():
    # The other shape the same mistake takes: an inline call such as
    # `lsfusion_get_guidance(article='Actions')`.
    tools = _tools()
    for name, tool in tools.items():
        for callee, args in re.findall(r"`?(lsfusion_[a-z_]+)\(([^)]*)\)", tool.description):
            if callee in tools:
                passed = set(re.findall(r"([a-z_]+)\s*=", args))
                bad = passed - set(tools[callee].inputSchema["properties"])
                assert not bad, f"{name}: calls {callee} with {sorted(bad)}, which it does not take"


# --- nullability of the results ---------------------------------------------
#
# A result field central declares nullable comes as an explicit null (FastMCP
# dumps the model without exclude_none). The plugin's REST endpoint /lsfusion
# publishes its own output schema and hands the null on as it is (the IDE's MCP
# server leaves a null field out instead), so if that schema's type leaves null
# out, a client validating structuredContent drops the whole result: an article
# walk or a single-query search lost all its chunks, and a feedback report
# failed after it had been stored (lsfusion/plugin-idea#96). The checks above
# compare names, so they passed. The platform proxy declares no output schema
# for these tools.
def _admits_null(prop: dict) -> bool:
    t = prop.get("type")
    return t == "null" or (isinstance(t, list) and "null" in t) \
        or any(_admits_null(s) for s in prop.get("anyOf") or [])


def _nullable(model) -> set[str]:
    """Result fields of a model and its nested $defs that central may send as null."""
    schema = model.model_json_schema()
    return {name for m in [schema, *(schema.get("$defs") or {}).values()]
            for name, prop in (m.get("properties") or {}).items() if _admits_null(prop)}


RETRIEVE_OUT_NULLABLE = _nullable(RetrieveDocsOutput)
FEEDBACK_OUT_NULLABLE = _nullable(FeedbackOutput)


def _call_args(text: str, opening: str) -> str | None:
    """What a call has after `opening` (which ends inside its parentheses), up
    to its closing parenthesis. None when there is no such call."""
    if opening not in text:
        return None
    start = text.index(opening) + len(opening)
    end = _closing(text, start, "(", ")")
    return None if end is None else text[start:end]


def _code(text: str, literals: bool = True) -> str:
    """The text without its comments, and without its string and char literals
    as well unless `literals`."""
    out, i = [], 0
    while i < len(text):
        j = _skip_literal(text, i)
        if j == i:
            out.append(text[i])
            i += 1
        else:
            if literals and not text.startswith(("//", "/*"), i):
                out.append(text[i:j])
            i = j
    return "".join(out)


def _declares_null(java: str, value: str | None, followed: frozenset = frozenset()) -> bool:
    """The value's schema admits null as central declares it - a null branch of
    its anyOf, or null in its type array - by itself or through a helper of the
    file it calls (in code, not in a string), followed through helpers that call
    helpers."""
    if value is None:
        return False
    code = _code(value)
    flat = re.sub(r"\s+", "", code)  # one shape however it is wrapped
    any_of, typed = _call_args(flat, '.put("anyOf",'), _call_args(flat, '.put("type",')
    if (any_of is not None and '.put(newJSONObject().put("type","null"))' in any_of) \
            or (typed is not None and typed.startswith("newJSONArray()") and '.put("null")' in typed):
        return True
    return any(f"JSONObject {h}(" in java
               and _declares_null(java, _brace_slice(java, f"JSONObject {h}("), followed | {h})
               for h in set(re.findall(r"\b([a-z]\w*)\(", _code(value, literals=False))) - followed)


def _kotlin_types(params: str) -> dict[str, str]:
    """The declared type of each `val` in a constructor's parameter list, by its
    wire name: its @SerialName, else its own."""
    return {wire: type_ for wire, (type_, _) in _kotlin_vals(params).items()}


def _kotlin_vals(params: str) -> dict[str, tuple[str, str | None]]:
    """(type, default or None) of each `val` in a constructor's parameter list,
    by its wire name: its @SerialName, else its own."""
    vals, depth, start, i = {}, 0, 0, 0
    pieces = []
    while i < len(params):
        j = _skip_literal(params, i)
        if j != i:
            i = j
            continue
        c = params[i]
        if c in "(<[{":
            depth += 1
        elif c in ")>]}":
            depth -= 1
        elif c == "," and depth == 0:
            pieces.append(params[start:i])
            start = i + 1
        i += 1
    pieces.append(params[start:])
    for piece in pieces:
        piece = _code(piece)
        val = re.search(r"\bval\s+(\w+)\s*:", piece)
        if val:
            wire = re.search(r'@SerialName\("([^"]+)"\)', piece[:val.start()])
            type_, eq, default = piece[val.end():].partition("=")
            vals[wire.group(1) if wire else val.group(1)] = (type_.strip(), default.strip() if eq else None)
    return vals


@siblings
@released
@pytest.mark.parametrize("descriptor,result,fields", [
    ("JSONObject buildRetrieveDocsToolDescriptor(", "JSONObject docItemSchema", RETRIEVE_OUT_NULLABLE),
    ("JSONObject buildReportFeedbackToolDescriptor(", "JSONObject outputSchema", FEEDBACK_OUT_NULLABLE),
], ids=["retrieve_docs", "report_feedback"])
def test_plugin_java_result_schemas_admit_null_where_central_sends_it(descriptor, result, fields):
    java = _need(PLUGIN_JAVA)
    region = _brace_slice(java, descriptor)
    output = _code(region[region.index(result):])  # the result schema, past the input one
    bad = sorted(f for f in fields if not _declares_null(java, _call_args(output, f'.put("{f}",')))
    assert not bad, f"plugin McpBaseService: result fields central sends as null, declared without null: {bad}"


@siblings
@released
@pytest.mark.parametrize("dto,fields", [
    ("data class RemoteDocItem(", RETRIEVE_OUT_NULLABLE),
    ("data class FeedbackOutput(", FEEDBACK_OUT_NULLABLE),
], ids=["retrieve_docs", "report_feedback"])
def test_plugin_kotlin_result_dtos_are_nullable_where_central_sends_null(dto, fields):
    # The IDE's MCP server builds the result schema from the DTO: a nullable
    # Kotlin type becomes ["<type>", "null"], a non-nullable one leaves null out.
    params = _call_args(_need(PLUGIN_KT), dto)
    assert params is not None, f"plugin McpToolset: no {dto}"
    types = _kotlin_types(params)
    bad = sorted(f for f in fields if not types.get(f, "").endswith("?"))
    assert not bad, f"plugin McpToolset {dto}: result fields central sends as null, not nullable: {bad}"


def test_central_nullable_results_guard():
    # An empty set would turn the two checks above green without checking anything.
    assert {"score", "query"} <= RETRIEVE_OUT_NULLABLE
    assert {"dedup_fingerprint", "detail"} <= FEEDBACK_OUT_NULLABLE


@siblings
@pytest.mark.parametrize("method", ["retrieveDocsDescriptor", "reportFeedbackDescriptor"],
                         ids=["retrieve_docs", "report_feedback"])
def test_platform_declares_no_result_schema_for_these_tools(method):
    # The platform has no nullability check like the plugin's only because it
    # declares no result schema for these tools.
    region = _code(_brace_slice(_need(PLATFORM), f"JSONObject {method}("))
    assert '"outputSchema"' not in region, \
        f"platform MCPDispatcher.{method} declares a result schema now: check its nulls as the plugin's are"


# --- what a released plugin depends on ----------------------------------------
#
# Everything above holds the server to the plugin's MASTER, the code its next
# release will ship. But mcp deploys on every commit, while a plugin release
# waits for review and users update when they update: the server is talking to
# plugins that are already out. The checks marked `released` hold it to what
# the latest released plugin depends on, and CI runs them against that
# release's tag (`v<version>`) as well as against master. They are narrower
# than the checks above on purpose: a released plugin that lacks a new
# parameter, or words a description the old way, still works; these catch what
# stops it working:
#
# - a tool it proxies is gone;
# - it sends an argument the server rejects: FastMCP refuses an unexpected
#   keyword, and its validation an enum value it does not know (an unknown field
#   inside the report is ignored, so only the tool's own arguments count);
# - the server requires a field the plugin cannot send;
# - the server sends a result the plugin's schema rejects: a field a closed
#   object does not declare, an enum value it does not list, a null where it
#   declares none (the two nullability checks above), or no field it requires.
#
# The REST endpoint /lsfusion passes arguments and results through as they are,
# under the schemas McpBaseService.java publishes; the IDE's MCP server builds
# its schemas from McpToolset.kt, which sends what its wrappers `put` and
# decodes the result into its DTOs.
_RELEASED_TOOLS = [
    ("lsfusion_retrieve_docs", "JSONObject buildRetrieveDocsToolDescriptor(", "suspend fun retrieveDocs("),
    ("lsfusion_get_guidance", "JSONObject buildGetGuidanceToolDescriptor(", "suspend fun getGuidance("),
    ("lsfusion_report_feedback", "JSONObject buildReportFeedbackToolDescriptor(", "suspend fun reportFeedback("),
]
_RELEASED_IDS = [t.removeprefix("lsfusion_") for t, _, _ in _RELEASED_TOOLS]
# Where the Kotlin DTOs that make up a tool's input are, for the fields nested
# in it; a tool missing here takes no nested input.
_KOTLIN_INPUT_DTOS = {"lsfusion_report_feedback": ("data class FbExpectation", "class McpToolset")}
# The result objects: (descriptor, the Java object's statement, tool, its
# central $defs entry or None for the result itself, the Kotlin DTO).
_RESULT_OBJECTS = [
    ("JSONObject buildRetrieveDocsToolDescriptor(", "JSONObject outputSchema =",
     "lsfusion_retrieve_docs", None, "data class RetrieveDocsOutput("),
    ("JSONObject buildRetrieveDocsToolDescriptor(", "JSONObject docItemSchema =",
     "lsfusion_retrieve_docs", "DocItem", "data class RemoteDocItem("),
    ("JSONObject buildReportFeedbackToolDescriptor(", "JSONObject outputSchema =",
     "lsfusion_report_feedback", None, "data class FeedbackOutput("),
]
_RESULT_IDS = ["retrieve_docs", "retrieve_docs-item", "report_feedback"]
_PUT = re.compile(r'\.put\(\s*"([^"]+)"\s*,')


def _statement(text: str, opening: str) -> str | None:
    """From `opening` to the `;` that ends its statement, past literals and
    brackets. None when there is no such statement."""
    if opening not in text:
        return None
    start = i = text.index(opening)
    depth = 0
    while i < len(text):
        j = _skip_literal(text, i)
        if j != i:
            i = j
            continue
        if text[i] in "([{":
            depth += 1
        elif text[i] in ")]}":
            depth -= 1
        elif text[i] == ";" and depth == 0:
            return text[start:i]
        i += 1
    return text[start:]


def _put_args(chain: str) -> dict[str, str]:
    """The value of each `.put("<key>", <value>)` a builder chain makes at its
    own level, by key: not the keys of the objects nested in those values."""
    out, depth, i = {}, 0, 0
    while i < len(chain):
        m = _PUT.match(chain, i) if depth == 0 else None
        if m:
            end = _closing(chain, m.end(), "(", ")")
            out[m.group(1)] = chain[m.end():end]
            i = len(chain) if end is None else end + 1
            continue
        j = _skip_literal(chain, i)
        if j != i:
            i = j
            continue
        if chain[i] in "([{":
            depth += 1
        elif chain[i] in ")]}":
            depth -= 1
        i += 1
    return out


def _literals(code: str) -> set[str]:
    return set(re.findall(r'"((?:[^"\\]|\\.)*)"', code))


def _offered_values(java: str) -> set[str]:
    """Enum values Java schema code offers: the literals in its `feArr(...)` and
    `.put("enum", ...)` arguments, apart from its `required` lists."""
    code, required = _code(java), '.put("required",'
    while (args := _call_args(code, required)) is not None:
        start = code.index(required)
        code = code[:start] + code[start + len(required) + len(args) + 1:]
    values: set[str] = set()
    for opening in ("feArr(", '.put("enum",'):
        for m in re.finditer(re.escape(opening), code):
            values |= _literals(_call_args(code[m.start():], opening) or "")
    return values


def _schema_lists(schema, key: str) -> set[str]:
    """Every string in every `key` list (`enum`, `required`) anywhere in a JSON schema."""
    if isinstance(schema, list):
        return set().union(*(_schema_lists(s, key) for s in schema))
    if not isinstance(schema, dict):
        return set()
    own = schema.get(key)
    out = {v for v in own if isinstance(v, str)} if isinstance(own, list) else set()
    return out.union(*(_schema_lists(v, key) for v in schema.values()))


def _java_input(java: str, descriptor: str) -> str:
    """A descriptor's code up to the end of its input schema: with the objects
    that schema is built from, without the result schema."""
    region = _brace_slice(java, descriptor)
    stmt = _statement(region, "JSONObject inputSchema =")
    assert stmt is not None, f"plugin McpBaseService: {descriptor} builds no inputSchema"
    return _code(region[:region.index(stmt) + len(stmt)])


def _java_object(java: str, descriptor: str, var: str) -> dict[str, str]:
    stmt = _statement(_brace_slice(java, descriptor), var)
    assert stmt is not None, f"plugin McpBaseService: {descriptor} has no {var!r}"
    return _put_args(_code(stmt))


def _java_properties(obj: dict[str, str], who: str) -> dict[str, str]:
    assert "properties" in obj, f"{who}: no properties found"
    return _put_args(obj["properties"])


def _kotlin_wire_keys(kotlin: str, wrapper: str) -> set[str]:
    """The keys a Kotlin wrapper puts into the arguments it sends."""
    region = _brace_slice(kotlin, wrapper)
    params = _call_args(region, wrapper)
    assert params is not None, f"plugin McpToolset: no {wrapper}"
    return set(_PUT.findall(_code(region[len(wrapper) + len(params):])))


def _kotlin_declared(region: str) -> set[str]:
    code = _code(region)
    return set(re.findall(r'@SerialName\("([^"]+)"\)', code)) | set(re.findall(r"\bval\s+(\w+)\s*:", code))


def _central_result(tool: str, def_name: str | None) -> dict:
    schema = _tools()[tool].outputSchema
    return schema["$defs"][def_name] if def_name else schema


def _problems(found: dict[str, set]) -> str:
    """`<who> [names]` for each non-empty set, joined; empty when all are."""
    return "; ".join(f"{who} {sorted(names)}" for who, names in found.items() if names)


@siblings
@released
def test_plugin_proxies_only_tools_the_server_has():
    java, kotlin = _need(PLUGIN_JAVA), _need(PLUGIN_KT)
    constants = dict(re.findall(r'static\s+final\s+String\s+(\w+)\s*=\s*"(lsfusion_\w+)"', java))
    listed = _call_args(_code(java), "REMOTE_TOOL_NAMES = Set.of(")
    assert listed is not None, "plugin McpBaseService: no REMOTE_TOOL_NAMES"
    proxied = {constants.get(n, n) for n in re.findall(r"\w+", listed)} \
        | set(re.findall(r'callRemoteToolResultJson\(\s*"(\w+)"', _code(kotlin)))
    assert proxied, "plugin: no proxied tools found"
    gone = sorted(proxied - set(_tools()))
    assert not gone, f"plugin proxies tools the server no longer has: {gone}"
    unchecked = sorted(proxied - {t for t, _, _ in _RELEASED_TOOLS})
    assert not unchecked, f"plugin proxies tools these checks do not cover, add them to _RELEASED_TOOLS: {unchecked}"


@siblings
@released
@pytest.mark.parametrize("tool,descriptor,wrapper", _RELEASED_TOOLS, ids=_RELEASED_IDS)
def test_plugin_sends_only_arguments_the_server_takes(tool, descriptor, wrapper):
    takes = set(_tools()[tool].inputSchema["properties"])
    java = set(_java_properties(_java_object(_need(PLUGIN_JAVA), descriptor, "JSONObject inputSchema ="),
                                f"plugin McpBaseService {descriptor}"))
    kotlin = _kotlin_wire_keys(_need(PLUGIN_KT), wrapper)
    assert java and kotlin, f"plugin: no arguments found for {tool}"
    bad = _problems({"McpBaseService offers": java - takes, "McpToolset sends": kotlin - takes})
    assert not bad, f"{tool}: arguments the server rejects: {bad}"


@siblings
@released
@pytest.mark.parametrize("tool,descriptor", [(t, d) for t, d, _ in _RELEASED_TOOLS if t != "lsfusion_get_guidance"],
                         ids=[i for i in _RELEASED_IDS if i != "get_guidance"])
def test_plugin_offers_only_enum_values_the_server_accepts(tool, descriptor):
    # The Kotlin DTOs type these as plain strings, so only the Java schemas offer values.
    offered = _offered_values(_java_input(_need(PLUGIN_JAVA), descriptor))
    assert offered, f"plugin McpBaseService {descriptor}: no enum values found"  # the parser still sees them
    bad = sorted(offered - _schema_lists(_tools()[tool].inputSchema, "enum"))
    assert not bad, f"plugin McpBaseService ({tool}) offers values the server rejects: {bad}"


@siblings
@released
@pytest.mark.parametrize("tool,descriptor,wrapper", _RELEASED_TOOLS, ids=_RELEASED_IDS)
def test_plugin_declares_every_field_the_server_requires(tool, descriptor, wrapper):
    schema = _tools()[tool].inputSchema
    top, nested = set(schema.get("required") or []), _schema_lists(schema.get("$defs") or {}, "required")
    java = _java_input(_need(PLUGIN_JAVA), descriptor)
    kotlin = _need(PLUGIN_KT)
    dtos = set()
    if nested:
        assert tool in _KOTLIN_INPUT_DTOS, f"{tool} takes nested input now: add its Kotlin DTOs to _KOTLIN_INPUT_DTOS"
        first, last = _KOTLIN_INPUT_DTOS[tool]
        dtos = _kotlin_declared(kotlin[kotlin.index(first):kotlin.index(last)])
    bad = _problems({
        "McpBaseService": {f for f in top | nested if '.put("%s"' % f not in java},
        f"McpToolset {wrapper}": top - _kotlin_wire_keys(kotlin, wrapper),
        "McpToolset DTOs": nested - dtos,
    })
    assert not bad, f"{tool}: fields the server requires, which the plugin does not declare: {bad}"


@siblings
@released
@pytest.mark.parametrize("descriptor,var,tool,def_name,dto", _RESULT_OBJECTS, ids=_RESULT_IDS)
def test_plugin_result_schema_lets_every_server_field_through(descriptor, var, tool, def_name, dto):
    obj = _java_object(_need(PLUGIN_JAVA), descriptor, var)
    if obj.get("additionalProperties", "").strip() != "false":
        return  # an open object takes any field
    who = f"plugin McpBaseService {descriptor} {var}"
    missing = sorted(set(_central_result(tool, def_name)["properties"]) - set(_java_properties(obj, who)))
    assert not missing, f"{who}: closed, so a result with these server fields fails validation: {missing}"


@siblings
@released
def test_plugin_kotlin_decodes_results_ignoring_fields_it_does_not_know():
    # What lets the Kotlin DTOs leave out a field the server sends.
    config = _code(_brace_slice(_need(PLUGIN_KT), "val json = Json"), literals=False)
    assert re.search(r"\bignoreUnknownKeys\s*=\s*true\b", config), \
        "plugin McpToolset: `json` must set ignoreUnknownKeys = true, or a new server field fails every decode"


@siblings
@released
@pytest.mark.parametrize("descriptor,var,tool,def_name,dto", _RESULT_OBJECTS, ids=_RESULT_IDS)
def test_plugin_result_enums_list_every_value_the_server_sends(descriptor, var, tool, def_name, dto):
    who = f"plugin McpBaseService {descriptor} {var}"
    props = _java_properties(_java_object(_need(PLUGIN_JAVA), descriptor, var), who)
    bad = {}
    for name, prop in _central_result(tool, def_name)["properties"].items():
        offered = _offered_values(props.get(name, ""))
        if offered and _enum_values(prop) - offered:  # no values listed: any string will do
            bad[name] = sorted(_enum_values(prop) - offered)
    assert not bad, f"{who}: values the server sends, which the schema does not list: {bad}"


@siblings
@released
@pytest.mark.parametrize("descriptor,var,tool,def_name,dto", _RESULT_OBJECTS, ids=_RESULT_IDS)
def test_plugin_requires_only_result_fields_the_server_sends(descriptor, var, tool, def_name, dto):
    # The server sends every field of its result models (nulls included), so a
    # field it has is a field that arrives.
    sends = set(_central_result(tool, def_name)["properties"])
    java = _literals(_java_object(_need(PLUGIN_JAVA), descriptor, var).get("required", ""))
    params = _call_args(_need(PLUGIN_KT), dto)
    assert params is not None, f"plugin McpToolset: no {dto}"
    kotlin = {w for w, (type_, default) in _kotlin_vals(params).items() if not type_.endswith("?") and default is None}
    assert java and kotlin, f"plugin: no required result fields found for {var} / {dto}"  # the parsers still see them
    bad = _problems({f"McpBaseService {var} requires": java - sends,
                       f"McpToolset {dto} cannot do without": kotlin - sends})
    assert not bad, f"{tool}: result fields the server does not send: {bad}"


def test_central_released_contract_guard():
    # The server side of the checks above, which pass for the wrong reason when
    # what they compare against is empty. Also a server change that breaks
    # every released plugin, whatever its version, and so one to make on
    # purpose: dropping a tool the plugin proxies.
    tools = _tools()
    assert {t for t, _, _ in _RELEASED_TOOLS} <= set(tools)
    fb = tools["lsfusion_report_feedback"]
    assert {"report", "signal_type", "primary_target"} <= \
        set(fb.inputSchema.get("required") or []) | _schema_lists(fb.inputSchema.get("$defs") or {}, "required")
    assert {"language", "paradigm", "how-to"} <= _schema_lists(tools["lsfusion_retrieve_docs"].inputSchema, "enum")
    assert {"recorded", "disabled", "rejected"} <= _enum_values(fb.outputSchema["properties"]["status"])
    assert {"docs", "status"} <= set(_central_result("lsfusion_retrieve_docs", None)["properties"])
    assert {"id", "source", "text"} <= set(_central_result("lsfusion_retrieve_docs", "DocItem")["properties"])
