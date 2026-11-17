"""Request-local MCP caller context, without starting a transport or server."""
from __future__ import annotations

import types

import pytest
from starlette.datastructures import Headers, QueryParams


@pytest.fixture
def caller_context():
    from mcp.server.lowlevel.server import request_ctx
    tokens = []

    def set_context(*, name="Claude Code", version="2.1.263", via=None, addons=(),
                    headers=None, request=True, session=True):
        info = types.SimpleNamespace(name=name, version=version)
        sess = types.SimpleNamespace(client_params=types.SimpleNamespace(clientInfo=info))
        req = types.SimpleNamespace(headers=Headers(headers or {}),
                                    query_params=QueryParams(([("via", via)] if via is not None else [])
                                                            + [("with", addon) for addon in addons]))
        ctx = types.SimpleNamespace(session=sess if session else None,
                                    request=req if request else None)
        tokens.append(request_ctx.set(ctx))
        return ctx

    yield set_context
    for token in reversed(tokens):
        request_ctx.reset(token)
