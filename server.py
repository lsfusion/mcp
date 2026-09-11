
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
from settings import BATCH_MAX_QUERIES
@mcp.tool(structured_output=True)
def lsfusion_retrieve_docs(
    query: Annotated[
        # The list branch carries the cap in the schema itself (maxItems), from
        # the same constant the runtime enforces, so a client sees the limit
        # before it sends six and learns about it from the error.
        str | Annotated[list[str], Field(max_length=BATCH_MAX_QUERIES)] | None,
        Field(description=f"One short technical question, or a list of at most {BATCH_MAX_QUERIES} DISTINCT queries for independent needs already known before this call — beyond that each one gets too small a share of one call's result budget to be worth asking, so split larger sets across calls. State what you need to learn or achieve; include a construct or module name when you know it, plus the behaviour or constraint that matters. Prefer `NEWSESSION APPLY behaviour when a nested session is canceled` to `sessions`: a bare noun is what every article in a branch is about, and the search answers it with whichever one is closest to that whole topic. Search is semantic, not literal — exact documentation wording is not required, and rephrasing beats retrying the same query. Batch only independent lookups, never alternative phrasings of one need; when one answer can determine or refine the next query, call again instead. In a batch, `type` and `exclude_ids` apply to every query, all queries share one result cap, a chunk answering two of them is returned once, and each result names the query it is credited to."),
    ] = None,
    type: Annotated[
        Literal["language", "paradigm", "how-to"] | None,
        Field(default=None, description="Optional branch filter; in a batch it applies to every query. Omit or pass null unless you specifically need answers from one branch. Choose by the ANSWER you need, not by the words in your query: `language` — the exact syntax, parameters and behaviour of a construct; `paradigm` — how the platform works: its concepts, its mechanisms and how they relate; `how-to` — a worked recipe for a task. An operator name in the query does not by itself justify `language`. For a mixed or uncertain need, omit the filter. `brief` and `rules` are not here at all: an area's capability map and its coding rules are read whole, by name, with `lsfusion_get_guidance`."),
    ] = None,
    exclude_ids: Annotated[
        list[str] | None,
        Field(default=None, description="Chunk `id` values you already hold. In a batch they apply to every query. They are excluded server-side BEFORE ranking, so the quota is spent on material you do not have. Use this to page deeper on the same information need. Do NOT use it to rephrase a query for a better ranking, or to ask a different question about the same area: the filter ignores the new query, so a chunk that is now the most relevant one would be dropped before ranking. Leave empty on the first call."),
    ] = None,
    article: Annotated[
        str | list[str] | None,
        Field(default=None, description="Name of the article to read instead of searching the whole corpus — one name, or a list of at most 12. Give it ALONE to WALK those articles from the top in DOCUMENT order; that is what you want when a chunk was on the right subject but plainly partial, because the constraint it depends on, the case it omits and the table it points at all sit elsewhere in the same article. Give it WITH `query` to SEARCH inside that set instead. Three shapes are accepted, all of which you are already holding: a published slug (`Interactive_view`); any chunk `id` verbatim (`Interactive_view::examples`) — everything from `::` names the section, the article is what gets read; or the DESTINATION of any `.md` link in chunk text (`../paradigm/Actions.md` and `Actions.md` both mean `Actions`) — the destination, never the visible label. Naming an article already picks its branch, so do not also pass `type`. Articles are PAGED, not delivered whole: every one named gets at least its next chunk, `status` says in words which of them are finished and which are not, and when any remain it gives you the exact call that continues — repeat it until `status` stops offering one."),
    ] = None,
) -> RetrieveDocsOutput:
    """Search official lsFusion documentation, or read one article of it by name. TWO SELECTORS, at least one required, and which you pass decides what you get: `query` alone searches all three branches by meaning and returns ranked chunks; `article` alone WALKS that one article from the top in document order — omit `query` to traverse; both together search inside that one article. Keeping `query` while adding `article` gets you a ranked search, not a traversal. Returns `{docs:[{id,source,text,score,query}], status}`; `status` is one sentence saying whether this response is the whole answer and, when it is not, the exact call that continues it. `score` is cosine similarity and is NULL in a traversal, which has nothing to be similar to; when it is null the list is in document order. `omitted` is how many matched but did not fit the response's size budget — non-zero means page with `exclude_ids`, zero means that was the whole answer. ROUTING: reach for `article` after a search, when a chunk is on the right subject but visibly partial — the constraint it depends on, the case it omits and the table it points at are elsewhere in the SAME article and invisible from inside the chunk. Its name is already in front of you: the part of the chunk `id` before `::`, or the basename of any `.md` link in the text. Searching again with different wording is the move this replaces. Omit `type` by default: each branch has its own share of the response, so setting `type` only removes the other two branches — it does not improve the branch it keeps — and naming an article picks its branch already. For capability maps and coding rules use `lsfusion_get_guidance`, which returns whole articles; reading the rules of an area you are about to work in is mandatory. The corpus is English-only (`docs/en/`) — cross-lingual embeddings make non-English queries work, but English wording gives the best recall."""
    return retrieve_docs_tool(query, type, exclude_ids, article)


import time
from tools.guidance import stamped_guidance, read_article, _log_guidance
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
        Field(default=None, description="Name of the `rules` area whose article you need — the short name in the FIRST COLUMN of the map inside the top `rules` article, not a slug (`Rules_logic`) and not a title. The whole article comes back: no search, no ranking, no excerpt. An area's article carries the current constraints and prescribed practices of that area — the traps accepted without a diagnostic that still change behaviour, the performance and structural choices already made, and the procedures whose order matters — and it is the authoritative source for them, so it is read rather than reconstructed from general lsFusion knowledge. Reading it is BINDING wherever the map states a trigger for it — the map's one-line summary is an index entry, not the rule, and an area you did not fetch is not an area without rules. Its silence is not evidence either: that an article states no rule about a construct does not make the construct valid, supported or safe. Omit BOTH parameters to get the top article of each branch, which is the start-of-session call and the only way to obtain the maps."),
    ] = None,
    brief: Annotated[
        str | None,
        Field(default=None, description="Name of the `brief` area whose article you need — the short name from the map inside the top `brief` article. Same shape as `rules`, and only one of the two may be given per call: one call delivers one whole article. Read an area's brief when the material already present does not identify a likely platform mechanism for the job — it is what stops you inventing a mechanism the platform already has. It is a survey, not an inventory: an article arrives whole, but a capability it does not mention is UNKNOWN, not absent, and that silence never supports a claim that lsFusion lacks something. Search `language` / `paradigm` / `how-to` with `lsfusion_retrieve_docs` before reporting that no documented mechanism exists. And the brief says WHAT exists; those three branches say how to write it."),
    ] = None,
) -> str:
    """Read ONE lsFusion article WHOLE — the coding rules of an area (`rules`), its capability map (`brief`), or any reference article by name (`article`). Naming beats searching whenever you already know WHICH article you need: you receive all of it, so nothing relevant can be silently withheld the way a top-N chunk retrieval withholds it. `rules` and `brief` are a small hierarchy of articles rather than a search corpus: they are not searchable at all. Call with NO arguments at the start of any lsFusion task: that returns the top article of both branches, each carrying the base material plus the complete map of its branch, and the `rules` map states per area the point at which reading that area's article stops being optional. ROUTING for `brief`: read an area's brief when the task describes an outcome and nothing already in hand names a likely lsFusion construct for it — that is what stops a mechanism being reinvented. When candidate constructs are already named, use `lsfusion_retrieve_docs` to assess them (`paradigm` for how they differ, `how-to` for what the usual scenario picks), and read the area's brief if none looks suitable. ROUTING for `article`: reach for it AFTER a search, when a returned chunk is on the right subject but visibly partial — a chunk cannot show you the constraint it depends on, the case it omits, or the table it points at, because those sit elsewhere in the same article. Pass that chunk's `id` unchanged. Searching again with different wording is the move this replaces. Apply each rule at its stated strength (MUST / MUST NOT are binding; SHOULD / SHOULD NOT are recommendations). Finding out WHICH article you need is a different tool: `lsfusion_retrieve_docs` searches `language`, `paradigm` and `how-to` by meaning. Every article is fenced by `=== BEGIN ... ===` / `=== END ... ===`; the END fence is what proves you are holding the complete text, so if it is missing — or your client saved the result to a file and showed you a preview — read the full file before using anything from it."""
    given = [n for n, v in (("rules", rules), ("brief", brief)) if v is not None]
    if len(given) > 1:
        # Still one call of the tool, so still one event — otherwise the
        # commonest misuse would be the one thing the adoption numbers miss.
        _log_guidance("both", None, "error", time.monotonic(),
                      error=ValueError(f"{' and '.join(given)} given together"))
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
