
from __future__ import annotations
import os
from typing import Annotated, Literal

from pydantic import Field

# FastMCP implements both stdio and Streamable HTTP transports
from mcp.server.fastmcp import FastMCP

from tools.guidance import build_instructions, stamped_guidance

# Deliver the lsFusion guidance at the MCP handshake. FastMCP returns
# `instructions` in the `initialize` response, and clients (e.g. Claude Code)
# embed it in the system prompt — so the Brief/Rules are guaranteed in context
# every session, surviving compaction, with no tool call required.
#
# Fetched ONCE at boot into GUIDANCE_SNAPSHOT and reused by both the handshake
# and the `lsfusion_get_guidance` tool, so both channels serve byte-identical
# content and the same version marker (refresh model = restart the server, per
# the deploy pipeline). A docs outage must not stop the server from starting:
# on failure GUIDANCE_SNAPSHOT stays None, the handshake degrades to a pointer,
# and the tool falls back to a live fetch so it can still recover mid-run.
try:
    GUIDANCE_SNAPSHOT: str | None = stamped_guidance()
    INSTRUCTIONS = build_instructions(GUIDANCE_SNAPSHOT)
except Exception as exc:  # noqa: BLE001 — any fetch failure degrades to the pointer
    GUIDANCE_SNAPSHOT = None
    INSTRUCTIONS = (
        "lsFusion guidance could not be fetched at connection time "
        f"({type(exc).__name__}). Call `lsfusion_get_guidance` and follow the "
        "rules it returns before working on any lsFusion task."
    )

# === Initialize MCP server ===
mcp = FastMCP(
    name="lsfusion-mcp",
    instructions=INSTRUCTIONS,
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
        Field(description="Short topical phrase. Semantic match (not literal); rephrase rather than retry the same query if results are weak."),
    ],
    type: Annotated[
        Literal["language", "paradigm", "how-to", "brief", "rules"] | None,
        Field(default=None, description="Optional sourceType filter (the docs folder). Omit (or pass null) to search all branches and merge. `language` = syntax / operator reference; `paradigm` = concepts / abstractions; `how-to` = task recipes; `brief` = concise capability map; `rules` = code conventions."),
    ] = None,
) -> RetrieveDocsOutput:
    """Search official lsFusion documentation (language, paradigm, how-to, brief, rules) for chunks relevant to a query. Returns `{docs:[{source,text,score}]}` sorted by descending score. Use `type` to narrow to one branch when known; omit to search all and merge. The corpus is English-only (`docs/en/`) — cross-lingual embeddings make non-English queries work, but English wording gives the best recall."""
    return retrieve_docs_tool(query, type)


@mcp.tool()
def lsfusion_get_guidance() -> str:
    """Fetch the brief overview and mandatory rules for working with lsFusion. The assistant MUST call this at the start of ANY lsFusion-related task if the guidance isn't already in context, and MUST then read and strictly follow all rules it returns. Normally the guidance is already delivered via the MCP handshake `instructions`, so call this only when that block is absent. The result carries a version marker identical to the handshake copy (both come from the same boot-time snapshot)."""
    # Serve the boot snapshot so this matches the handshake byte-for-byte; only
    # if the boot fetch failed do we attempt a live fetch to recover mid-run.
    return GUIDANCE_SNAPSHOT if GUIDANCE_SNAPSHOT is not None else stamped_guidance()


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
