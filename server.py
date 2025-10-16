
from __future__ import annotations
from typing import List, Dict, Any
import os

# FastMCP implements both stdio and Streamable HTTP transports
from mcp.server.fastmcp import FastMCP

# Import tools; keep this file minimal so you can add more tools later
from tools.rag_retrieve import retrieve_docs


# === Initialize MCP server ===
mcp = FastMCP(
    name="lsfusion-mcp",
    host=os.getenv("MCP_HOST", "0.0.0.0"),
    port=int(os.getenv("MCP_PORT", "8000")),
    streamable_http_path="/mcp",
    sse_path="/mcp/sse",
)


# === Tool: retrieve_docs ===
@mcp.tool()
def retrieve_docs_tool(query: str) -> List[Dict[str, Any]]:
    """
    Fetch prioritized chunks from your RAG store—documentation, how-tos, tutorials and articles—
    based on a single search query.
    """
    return retrieve_docs(query=query)


# Template for future tools:
# @mcp.tool()
# def lint_code(language: str, code: str) -> dict:
#     """Run a basic syntax check or lint for the given language."""
#     return {"language": language, "issues": []}


if __name__ == "__main__":
    # transport = os.getenv("MCP_TRANSPORT", "stdio")
    # if transport == "stdio":
    #     mcp.run("stdio")
    # else:
    mcp.run("streamable-http")
