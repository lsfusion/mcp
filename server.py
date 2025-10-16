
from __future__ import annotations
from typing import List, Dict, Any

# FastMCP implements both stdio and Streamable HTTP transports
from mcp.server.fastmcp import FastMCP

# Import tools; keep this file minimal so you can add more tools later
from tools.rag_retrieve import retrieve_docs

# A generic server name to host multiple tools
mcp = FastMCP("lsfusion-mcp")


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
    import argparse

# FastMCP reads MCP_HOST and MCP_PORT from environment for HTTP transport

    parser = argparse.ArgumentParser(description="MCP server for lsfusion tools")
    parser.add_argument("transport", choices=["stdio", "http"], nargs="?", default="stdio")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.transport == "stdio":
        # Local development: MCP over stdio
        mcp.run("stdio")
    else:
        # Production HTTP transport (host/port from MCP_HOST and MCP_PORT env vars)
        mcp.run("sse")
