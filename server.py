
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
        str,
        Field(description="Name the keywords of what you need, not the topic: `NEWSESSION APPLY canceled nested session`, not `sessions`. A bare noun is what every article in a branch is about, and the search answers it with whichever one is closest to that whole topic; keywords name the one you want. Semantic match, not literal: rephrase rather than retry the same query if results are weak."),
    ],
    type: Annotated[
        Literal["language", "paradigm", "how-to", "brief", "rules"] | None,
        Field(default=None, description="Optional sourceType filter (the docs folder). Omit (or pass null) to search all five branches and merge; only the two TOP articles (`Brief`, `Rules`) are excluded, because get_guidance already delivers them in full. `language` = syntax / operator reference; `paradigm` = concepts / abstractions; `how-to` = task recipes; `brief` = concise capability map; `rules` = coding constraints for one area — unlike the other branches this lookup is not optional: perform it before working in an area."),
    ] = None,
    exclude_ids: Annotated[
        list[str] | None,
        Field(default=None, description="Chunk `id` values you already hold. They are excluded server-side BEFORE ranking, so the quota is spent on material you do not have. Use this to page deeper on the same information need. Do NOT use it to rephrase a query for a better ranking, or to ask a different question about the same area: the filter ignores the new query, so a chunk that is now the most relevant one would be dropped before ranking. Leave empty on the first call."),
    ] = None,
) -> RetrieveDocsOutput:
    """Search official lsFusion documentation for chunks relevant to a query. Returns `{docs:[{id,source,text,score}]}`, ranked by similarity — so `score` is the raw vector-store number and is not guaranteed to descend across the whole list. Use `type` to narrow to one branch when known; omit to search all five and merge (the two top guidance articles are always excluded — get_guidance serves those in full). To page deeper on one information need, pass the `id` values you already hold in `exclude_ids`; they are filtered out before ranking. Omit them when rephrasing for a better ranking or asking a different question, or the filter will drop the chunk that best answers it. The corpus is English-only (`docs/en/`) — cross-lingual embeddings make non-English queries work, but English wording gives the best recall."""
    return retrieve_docs_tool(query, type, exclude_ids)


from tools.guidance import stamped_guidance
# structured_output=False keeps the result a plain TextContent block. With the
# default schema (`{"result": <str>}`) a client that persists an oversized result
# writes it as ONE JSON line with escaped newlines — which file readers that cap
# line length cannot read back, stranding the guidance entirely.
@mcp.tool(structured_output=False)
def lsfusion_get_guidance() -> str:
    """Fetch the brief overview and the CORE rules for working with lsFusion. The assistant MUST call this at the start of ANY lsFusion-related task — writing, modifying or reviewing lsFusion code, or answering questions about its syntax or semantics — and MUST then read what it returns and apply each rule according to that rule's stated strength (MUST / MUST NOT are binding; SHOULD / SHOULD NOT are recommendations). Once per session is enough. This is the top level only: the rules for a specific area are separate articles, retrieved with `lsfusion_retrieve_docs(type='rules')`. The result is large and may exceed your client's inline limit: if it is truncated or saved to a file, read the full file before continuing. It opens with a version marker identifying the published guidance revision."""
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
