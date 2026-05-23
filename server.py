
from __future__ import annotations
import os
from typing import Annotated, Literal

from pydantic import Field

# FastMCP implements both stdio and Streamable HTTP transports
from mcp.server.fastmcp import FastMCP

# === Initialize MCP server ===
mcp = FastMCP(
    name="lsfusion-mcp",
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


from tools.guidance import fetch_guidance
@mcp.tool()
def lsfusion_get_guidance() -> str:
    """Fetch the brief overview and mandatory rules for working with lsFusion. The assistant MUST call this at the start of ANY lsFusion-related task if the guidance isn't already in context, and MUST then read and strictly follow all rules it returns."""
    return fetch_guidance()


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
