"""Who is calling — as far as the protocol will say.

The MODEL is not knowable here, and not by any supported route: MCP carries no
model field anywhere (`initialize` gives `clientInfo{name,version}`, `tools/call`
gives `{name,arguments}`), and `sampling` — which does return a model name — is
server-to-client, arrives only after inference, and is deprecated as of the
2026-07-28 revision. So four different models behind one client are
indistinguishable from this side, and a log field claiming otherwise would be a
field that lies.

What IS knowable is the client: `clientInfo` from the handshake, and the
User-Agent of the HTTP request. That answers "Claude Code or Cursor or the IDE
plugin", not "Opus or Fable". Recorded because it is honest and free, not
because it answers the question that prompted it.

`x-agent-model` is read if a caller sets it: nothing sends it today, and Claude
Code's own header config is reported not to reach `tools/call` at all, but a
harness comparing models can set it on a proxy, and reading it costs one dict
lookup. Absent it, the field is simply missing rather than guessed.
"""
from __future__ import annotations

_MODEL_HEADER = "x-agent-model"


def caller_fields() -> dict:
    """Client identity for an event. NEVER raises; returns {} when unavailable
    (stdio transport, no active request, a future SDK that moves any of this)."""
    out: dict = {}
    try:
        from mcp.server.lowlevel.server import request_ctx
        ctx = request_ctx.get(None)
        if ctx is None:
            return out
        try:
            info = ctx.session.client_params.clientInfo
            if info is not None:
                if info.name:
                    out["client"] = str(info.name)[:60]
                if info.version:
                    out["client_version"] = str(info.version)[:40]
        except Exception:  # noqa: BLE001 — identity is never worth an exception
            pass
        try:
            headers = ctx.request.headers  # Starlette request on HTTP transports
            ua = headers.get("user-agent")
            if ua:
                out["user_agent"] = str(ua)[:80]
            declared = headers.get(_MODEL_HEADER)
            if declared:
                # Caller-declared, never verified. Named so no reader mistakes
                # it for something the protocol established.
                out["declared_model"] = str(declared)[:60]
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        return {}
    return out
