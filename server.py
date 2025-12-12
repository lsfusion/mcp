
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
    Fetch prioritized chunks from lsFusion RAG store (documentation and language reference) —
    based on a single search query.
    """
    return retrieve_docs_tool(query)


from tools.rag_retrieve import retrieve_samples_tool
@mcp.tool(structured_output=True)
def retrieve_samples(query: str) -> RetrieveDocsOutput:
    """
    Fetch prioritized chunks from lsFusion RAG store (how-tos and code samples) — based on a single search query.
    """
    return retrieve_samples_tool(query)


from tools.rag_retrieve import retrieve_learning_tool
@mcp.tool(structured_output=True)
def retrieve_learning(query: str) -> RetrieveDocsOutput:
    """
    Fetch prioritized chunks from lsFusion RAG store (tutorials and articles) — based on a single search query.
    """
    return retrieve_learning_tool(query)


from tools.validate_dsl import validate_dsl_statements_tool, DSLValidationResult
@mcp.tool(structured_output=True)
def validate_dsl_statements(text: str) -> DSLValidationResult:
    """
    Validate the syntax of the list of lsFusion statements
    """
    return validate_dsl_statements_tool(text)


@mcp.tool()
def get_brief() -> str:
    """
    Initialize context for this MCP server.

    Call this tool first before using any other tools.
    It loads the contents of brief.md so the calling model can read guidance
    before generating or editing lsFusion code.
    """
    brief_path = os.path.join(os.path.dirname(__file__), "brief.md")
    try:
        with open(brief_path, "r", encoding="utf-8") as brief_file:
            return brief_file.read()
    except FileNotFoundError:
        return "brief.md not found"
    except OSError as exc:
        return f"Unable to read brief.md: {exc}"


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
