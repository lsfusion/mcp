
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
def lsfusion_retrieve_docs(query: str) -> RetrieveDocsOutput:
    """
    Fetch prioritized chunks from lsFusion RAG store (official documentation and language reference) —
    based on a single search query.
    """
    return retrieve_docs_tool(query)


from tools.rag_retrieve import retrieve_samples_tool
@mcp.tool(structured_output=True)
def lsfusion_retrieve_howtos_and_samples(query: str) -> RetrieveDocsOutput:
    """
    Fetch prioritized chunks from lsFusion RAG store (code samples for combined tasks / scenarios and how-to) —
    based on a single search query.
    """
    return retrieve_samples_tool(query)


from tools.rag_retrieve import retrieve_learning_tool
@mcp.tool(structured_output=True)
def lsfusion_retrieve_community(query: str) -> RetrieveDocsOutput:
    """
    Fetch prioritized chunks from lsFusion RAG store (tutorials, articles, and community discussions) — based on a single search query.
    Use this ONLY for deep, ambiguous tasks when other retrieval tools (docs, howtos) did not provide a solution.
    """
    return retrieve_learning_tool(query)


from tools.validate_dsl import validate_dsl_statements_tool, DSLValidationResult
@mcp.tool(structured_output=True)
def lsfusion_validate_syntax(text: str) -> DSLValidationResult:
    """
    Validate the syntax of the list of lsFusion statements
    """
    return validate_dsl_statements_tool(text)


@mcp.tool()
def lsfusion_get_guidance() -> str:
    """
    Fetch the brief overview and mandatory rules for working with lsFusion.
    IMPORTANT: The assistant MUST call this tool before ANY task related to lsFusion if it's not already in your context.
    The assistant MUST read and strictly follow all rules and guidelines provided by this tool for ANY lsFusion-related task.
    """
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
