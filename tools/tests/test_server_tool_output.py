"""Guards the tool-result SHAPE that `lsfusion_get_guidance` presents to clients.

A `-> str` tool makes FastMCP synthesize an output schema (`{"result": <str>}`)
and attach `structuredContent`. A client that persists an oversized result then
writes it as ONE JSON line with escaped newlines, which file readers that cap
line length cannot read back — stranding the guidance with no way to recover it.
`structured_output=False` keeps the result a plain TextContent block instead.

That fix lives in a single keyword argument and would be silently undone by a
refactor, so assert the observable contract rather than the source.
"""

from __future__ import annotations

import asyncio
import importlib

import pytest


@pytest.fixture(scope="module")
def mcp_server():
    # rag_retrieve builds an OpenAI client at import time; a dummy key is enough
    # (nothing here makes a network call).
    import os

    os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy")
    return importlib.import_module("server").mcp


def _tool(mcp_server, name):
    tools = asyncio.run(mcp_server.list_tools())
    by_name = {t.name: t for t in tools}
    assert name in by_name, f"{name} not registered; got {sorted(by_name)}"
    return by_name[name]


def test_get_guidance_declares_no_output_schema(mcp_server):
    # An output schema is what drags structuredContent along.
    assert _tool(mcp_server, "lsfusion_get_guidance").outputSchema is None


def test_get_guidance_result_is_plain_text_not_json_wrapped(mcp_server, monkeypatch):
    import tools.guidance as g

    monkeypatch.setattr(g, "_fetch", lambda url, timeout: "line1\nline2")

    result = asyncio.run(
        mcp_server._tool_manager.call_tool(
            "lsfusion_get_guidance", {}, context=None, convert_result=True
        )
    )

    # A structured tool returns (content, structuredContent); a plain one returns
    # just the content blocks. Anything tuple-shaped means the wrapper is back.
    assert not isinstance(result, tuple), "structuredContent reintroduced"
    assert [b.type for b in result] == ["text"]
    text = result[0].text
    assert "line1\nline2" in text
    assert text.rstrip().endswith("===")  # the fence must survive the transport
    assert not text.lstrip().startswith('{"result"')


def test_get_guidance_serves_a_named_article_as_plain_text(mcp_server, monkeypatch):
    # Same guarantee for the by-name call: an article delivered as one escaped
    # JSON line is an article a file reader cannot read back.
    import tools.guidance as g

    monkeypatch.setattr(g, "read_article",
                        lambda branch, name, timeout=None: f"ARTICLE {branch}/{name}")
    import server as srv
    monkeypatch.setattr(srv, "read_article", g.read_article)

    result = asyncio.run(
        mcp_server._tool_manager.call_tool(
            "lsfusion_get_guidance", {"rules": "logic"}, context=None, convert_result=True
        )
    )

    assert not isinstance(result, tuple), "structuredContent reintroduced"
    assert [b.type for b in result] == ["text"]
    assert result[0].text == "ARTICLE rules/logic"


def test_retrieve_docs_keeps_its_structured_schema(mcp_server):
    # The plain-text treatment is specific to guidance; the RAG tool's typed
    # schema is intentional and must not be collaterally stripped.
    schema = _tool(mcp_server, "lsfusion_retrieve_docs").outputSchema
    assert schema and "docs" in schema["properties"]


def test_retrieve_docs_publishes_the_id_and_exclude_ids_contract(mcp_server):
    # `id` out and `exclude_ids` in are the whole point of the follow-up lookup:
    # if either falls out of the published schema, callers cannot use it and the
    # hand-mirrored proxies have nothing to mirror.
    tool = _tool(mcp_server, "lsfusion_retrieve_docs")
    assert "exclude_ids" in tool.inputSchema["properties"]

    defs = tool.outputSchema.get("$defs", {})
    doc_item = defs.get("DocItem") or {}
    assert "id" in doc_item.get("properties", {}), tool.outputSchema
