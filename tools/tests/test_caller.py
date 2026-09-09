"""Client identity on events — and the honesty constraint that shapes it.

The model is not knowable from MCP by any supported route, so nothing here may
produce a model field the protocol did not carry. What these tests pin is that
the client IS recorded, that a caller-declared model is labelled as declared,
and that identity can never break a call.
"""
from __future__ import annotations

import types

import pytest

from tools.caller import caller_fields


class _Headers(dict):
    def get(self, k, default=None):  # Starlette headers are case-insensitive
        return super().get(k.lower(), default)


def _ctx(*, name="Claude Code", version="2.1.263", headers=None, request=True, session=True):
    info = types.SimpleNamespace(name=name, version=version)
    sess = types.SimpleNamespace(client_params=types.SimpleNamespace(clientInfo=info))
    req = types.SimpleNamespace(headers=_Headers(headers or {}))
    return types.SimpleNamespace(session=sess if session else None,
                                 request=req if request else None)


@pytest.fixture
def set_ctx():
    # The real ContextVar: its `get` is read-only, so set a value and reset it.
    from mcp.server.lowlevel.server import request_ctx
    tokens = []

    def _set(ctx):
        tokens.append(request_ctx.set(ctx))
    yield _set
    for tok in reversed(tokens):
        request_ctx.reset(tok)


def test_records_the_client_and_its_user_agent(set_ctx):
    set_ctx(_ctx(headers={"user-agent": "Cursor/3.19.13 (win32 x64)"}))
    f = caller_fields()
    assert f["client"] == "Claude Code" and f["client_version"] == "2.1.263"
    assert f["user_agent"].startswith("Cursor/3.19.13")


def test_never_invents_a_model(set_ctx):
    # The whole point: no supported MCP route carries the model, so the field
    # must be ABSENT rather than guessed from the client or the user agent.
    set_ctx(_ctx(headers={"user-agent": "Claude Code/2.1.263"}))
    f = caller_fields()
    assert "model" not in f and "declared_model" not in f


def test_a_declared_model_is_labelled_as_declared(set_ctx):
    # Nothing sends this today. If a harness ever does, the name must keep
    # saying that a caller asserted it and the protocol did not.
    set_ctx(_ctx(headers={"x-agent-model": "claude-opus-5"}))
    assert caller_fields()["declared_model"] == "claude-opus-5"


def test_missing_request_or_session_degrades_to_partial(set_ctx):
    set_ctx(_ctx(request=False))                      # stdio transport: no HTTP request
    assert caller_fields() == {"client": "Claude Code", "client_version": "2.1.263"}
    set_ctx(_ctx(session=False, headers={"user-agent": "x"}))
    assert caller_fields() == {"user_agent": "x"}


def test_no_active_request_is_empty_not_an_error():
    # Nothing set: the default transport state outside a request.
    assert caller_fields() == {}


def test_identity_can_never_break_a_call(monkeypatch):
    import mcp.server.lowlevel.server as srv

    class _Exploding:
        def get(self, default=None):
            raise RuntimeError("boom")

    monkeypatch.setattr(srv, "request_ctx", _Exploding())
    assert caller_fields() == {}


def test_values_are_capped(set_ctx):
    set_ctx(_ctx(name="x" * 500, version="y" * 500,
                 headers={"user-agent": "u" * 500, "x-agent-model": "m" * 500}))
    f = caller_fields()
    assert len(f["client"]) <= 60 and len(f["client_version"]) <= 40
    assert len(f["user_agent"]) <= 80 and len(f["declared_model"]) <= 60
