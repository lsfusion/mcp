
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
        Literal["language", "paradigm"] | None,
        Field(default=None, description="Optional sourceType filter. Omit (or pass null) to search both axes. `language` returns syntax / operator reference chunks; `paradigm` returns conceptual / abstraction chunks."),
    ] = None,
) -> RetrieveDocsOutput:
    """Search official lsFusion documentation (language reference + paradigm concepts) for chunks relevant to a query. Returns `{docs:[{source,text,score}]}` sorted by descending score. Use `type` to narrow by axis when known; omit to search both. The corpus is English-only (`docs/en/`) — cross-lingual embeddings make non-English queries work, but English wording gives the best recall."""
    return retrieve_docs_tool(query, type)


@mcp.tool()
def lsfusion_get_guidance() -> str:
    """Fetch the brief overview and mandatory rules for working with lsFusion. The assistant MUST call this at the start of ANY lsFusion-related task if the guidance isn't already in context, and MUST then read and strictly follow all rules it returns."""
    base_dir = os.path.dirname(__file__)
    output = []

    for filename in ["brief.md", "rules.md"]:
        path = os.path.join(base_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                output.append(f"--- {filename} ---\n{f.read()}")
        except Exception as e:
            output.append(f"--- {filename} ---\nError reading file: {e}")

    return "\n\n".join(output)


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
