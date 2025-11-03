
from __future__ import annotations
import os

# FastMCP implements both stdio and Streamable HTTP transports
from mcp.server.fastmcp import FastMCP

# === Initialize MCP server ===
mcp = FastMCP(
    name="lsfusion-mcp",
    host=os.getenv("MCP_HOST", "0.0.0.0"),
    port=int(os.getenv("MCP_PORT", "8000")),
    streamable_http_path="/",
    sse_path="/sse",
)


# Import tools; keep this file minimal so you can add more tools later
from tools.rag_retrieve import retrieve_docs_tool, RetrieveDocsOutput
@mcp.tool(structured_output=True)
def retrieve_docs(query: str) -> RetrieveDocsOutput:
    """
    Fetch prioritized chunks from lsFusion RAG store (documentation, how-tos, tutorials and articles) —
    based on a single search query.
    """
    return retrieve_docs_tool(query)

from tools.validate_dsl import validate_dsl_statements_tool, DSLValidationResult
@mcp.tool(structured_output=True)
def validate_dsl_statements(text: str) -> DSLValidationResult:
    """
    Validate the syntax of the list of lsFusion statements
    """
    return validate_dsl_statements_tool(text)

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
