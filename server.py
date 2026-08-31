
from __future__ import annotations
import os
from typing import Annotated, Literal

from pydantic import Field

# FastMCP implements both stdio and Streamable HTTP transports
from mcp.server.fastmcp import FastMCP

from tools.guidance import SERVER_INSTRUCTIONS

# `instructions` is returned at the `initialize` handshake and clients embed it
# in the system prompt. It holds a POINTER to lsfusion_get_guidance, not the
# guidance itself: clients cap this field (Claude Code at ~2 KB), so shipping the
# ~50 KB body here delivers a silently mutilated copy with no recoverable tail.
# The tool is the only channel that can carry the whole thing.
#
# === Initialize MCP server ===
mcp = FastMCP(
    name="lsfusion-mcp",
    instructions=SERVER_INSTRUCTIONS,
    host=os.getenv("MCP_HOST", "0.0.0.0"),
    port=int(os.getenv("MCP_PORT", "8000")),
    streamable_http_path="/mcp",
    sse_path="/sse",
)


# Import tools; keep this file minimal so you can add more tools later
from tools.rag_retrieve import retrieve_docs_tool, RetrieveDocsOutput
@mcp.tool(structured_output=True)
def lsfusion_retrieve_docs(
    query: Annotated[
        str | list[str],
        Field(description="One short technical query, or a list of DISTINCT queries for independent needs already known before this call. Batch only lookups that do not depend on one another; when one answer can determine or refine the next query, call the tool again instead. Do not batch alternative phrasings of one need. In a batch, `type` and `exclude_ids` apply to every query, all queries share one result cap, a chunk answering two of them is returned once, and each result names the query it is credited to. Name the keywords of what you need, not the topic: `NEWSESSION APPLY canceled nested session`, not `sessions`. A bare noun is what every article in a branch is about, and the search answers it with whichever one is closest to that whole topic; keywords name the one you want. Semantic match, not literal: rephrase rather than retry the same query if results are weak."),
    ],
    type: Annotated[
        Literal["language", "paradigm", "how-to", "brief", "rules"] | None,
        Field(default=None, description="Optional sourceType filter (the docs folder); in a batch it applies to every query. Omit (or pass null) to search all three reference branches and merge. `language` = syntax / operator reference; `paradigm` = concepts / abstractions; `how-to` = task recipes. `brief` and `rules` are still accepted but no longer searched: they are read whole, by name, with `lsfusion_get_guidance` — passing one here returns a pointer to that tool, not documentation."),
    ] = None,
    exclude_ids: Annotated[
        list[str] | None,
        Field(default=None, description="Chunk `id` values you already hold. In a batch they apply to every query. They are excluded server-side BEFORE ranking, so the quota is spent on material you do not have. Use this to page deeper on the same information need. Do NOT use it to rephrase a query for a better ranking, or to ask a different question about the same area: the filter ignores the new query, so a chunk that is now the most relevant one would be dropped before ranking. Leave empty on the first call."),
    ] = None,
) -> RetrieveDocsOutput:
    """Search official lsFusion documentation for chunks relevant to a query. Returns `{docs:[{id,source,text,score,query}]}`, ranked by similarity — so `score` is the raw vector-store number and is not guaranteed to descend across the whole list. Use `type` to narrow to one branch when known; omit to search all three reference branches and merge. The `brief` and `rules` branches are NOT here — an area's capability map and its coding rules are read whole, by name, with `lsfusion_get_guidance`, and reading the rules of an area you are about to work in is mandatory. To page deeper on one information need, pass the `id` values you already hold in `exclude_ids`; they are filtered out before ranking. Omit them when rephrasing for a better ranking or asking a different question, or the filter will drop the chunk that best answers it. The corpus is English-only (`docs/en/`) — cross-lingual embeddings make non-English queries work, but English wording gives the best recall."""
    return retrieve_docs_tool(query, type, exclude_ids)


from tools.guidance import stamped_guidance, read_article
# structured_output=False keeps the result a plain TextContent block. Two
# reasons, and both bite here. With the default schema (`{"result": <str>}`) a
# client that persists an oversized result writes it as ONE JSON line with
# escaped newlines, which file readers that cap line length cannot read back —
# stranding the guidance entirely. And a client that receives `structuredContent`
# may prefer it and DISCARD the text blocks, re-serializing with JSON.stringify;
# a Markdown article delivered that way reaches the model as one escaped line.
@mcp.tool(structured_output=False)
def lsfusion_get_guidance(
    rules: Annotated[
        str | None,
        Field(default=None, description="Name of the `rules` area whose article you need — the short name in the FIRST COLUMN of the map inside the top `rules` article, not a slug (`Rules_logic`) and not a title. The whole article comes back: no search, no ranking, no excerpt. Reading an area's article is BINDING wherever the map states a trigger for it — the map's one-line summary is an index entry, not the rule, and an area you did not fetch is not an area without rules. Omit BOTH parameters to get the top article of each branch, which is the start-of-session call and the only way to obtain the maps."),
    ] = None,
    brief: Annotated[
        str | None,
        Field(default=None, description="Name of the `brief` area whose article you need — the short name from the map inside the top `brief` article. Same shape as `rules`, and only one of the two may be given per call: one call delivers one whole article. The `brief` branch is a capability map, so reading an area before working in it is STRONGLY RECOMMENDED rather than binding — it is what stops you inventing a mechanism the platform already has. It is not a substitute for `lsfusion_retrieve_docs`: the brief says WHAT exists, the `language` / `paradigm` / `how-to` branches say how to write it."),
    ] = None,
) -> str:
    """Read ONE lsFusion guidance article WHOLE — the coding rules of an area (`rules`) or its capability map (`brief`). These two branches are a small hierarchy of articles, not a search corpus: you name an article and receive all of it, so nothing relevant can be silently withheld the way a top-N chunk retrieval withholds it. Call with NO arguments at the start of any lsFusion task: that returns the top article of both branches, each carrying the base material plus the complete map of its branch, and the `rules` map states per area the point at which reading that area's article stops being optional. Apply each rule at its stated strength (MUST / MUST NOT are binding; SHOULD / SHOULD NOT are recommendations). Syntax, concepts and recipes are a different tool: `lsfusion_retrieve_docs`. Every article is fenced by `=== BEGIN ... ===` / `=== END ... ===`; the END fence is what proves you are holding the complete text, so if it is missing — or your client saved the result to a file and showed you a preview — read the full file before using anything from it."""
    if rules is not None and brief is not None:
        raise ValueError(
            "Pass either `rules` or `brief`, not both: one call delivers one whole "
            "article. Call twice, or omit both for the top article of each branch."
        )
    if rules is not None:
        return read_article("rules", rules)
    if brief is not None:
        return read_article("brief", brief)
    return stamped_guidance()


from tools.feedback import report_feedback_tool, FeedbackReport, FeedbackOutput
@mcp.tool(structured_output=True)
def lsfusion_report_feedback(report: FeedbackReport) -> FeedbackOutput:
    """Submit ONE anonymous, depersonalized reinforcement-quality signal so lsFusion docs / RAG / eval diagnostics / the platform can be improved. Use `signal_type` to say what kind: a documentation gap, an expectation-mismatch (you expected lsFusion to behave/mean X but it was actually Y — fill `expectation`), an unclear/unactionable `eval` error, a missing capability, a RAG miss, or other. Call this ONLY per the workflow rule from `lsfusion_get_guidance` (the friction was action-affecting) AND only after the user explicitly consents. Send NO source code, file paths, schema/table/customer names, or secrets — only the depersonalized journey (eval errors, the doc queries you tried, expected-vs-actual, how you resolved it) and a recommendation. The feedback is a suggestion, not a decision. Returns `{report_id, status, dedup_fingerprint}`."""
    return report_feedback_tool(report)


# Template for future tools:
# @mcp.tool()
# def lint_code(language: str, code: str) -> dict:
#     """Run a basic syntax check or lint for the given language."""
#     return {"language": language, "issues": []}


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "")
    if transport == "stdio":
        mcp.run()
    else:
        mcp.run("streamable-http")
