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

from dataclasses import dataclass

_MODEL_HEADER = "x-agent-model"

# Two hierarchies: `idedev` extends `dev` (development on the source; a direct
# connection is `dev` itself), `webapp` extends `app` (work in a running
# application). A kind's articles = its ancestor's, then its own. Old application
# proxies cannot tell app from webapp, so their fallback carries both.
_VIA_ARTICLES = {
    "dev": ("via_dev",),
    "idedev": ("via_dev", "via_idedev"),
    "app": ("via_app",),
    "webapp": ("via_app", "via_webapp"),
    "legacy-server": ("via_app", "via_webapp"),
}

@dataclass(frozen=True)
class CallerVia:
    kind: str
    articles: tuple[str, ...]
    given: str | None = None  # Only an unrecognized `via` value, capped for logging.


def _fallback_kind(ctx) -> str:
    """The kind of a request without `via`, from our old proxies' client names.

    clientInfo is the SESSION's: this fallback needs the stateful transport (the
    server's default) - a stateless call has none, and then it is `dev`.
    """
    try:
        name = ctx.session.client_params.clientInfo.name
    except Exception:  # noqa: BLE001 — no session, no fallback
        return "dev"
    if name == "lsfusion-intellij":
        return "idedev"
    if name == "lsfusion-server":
        return "legacy-server"
    return "dev"


def resolve_via() -> CallerVia:
    """Resolve the CURRENT HTTP request's access kind; never cache on a session.

    Only our old proxies' client names are fallbacks. No HTTP request (including
    stdio), or unavailable context, means `dev`. Identity must never break a
    call. Other query parameters - the skills plugin's `with=skills` among them -
    are not read: they carry no meaning for the answer yet.
    """
    kind = "dev"
    given = None
    try:
        from mcp.server.lowlevel.server import request_ctx
        ctx = request_ctx.get(None)
        if ctx is not None and ctx.request is not None:
            value = ctx.request.query_params.get("via")
            if value is not None:
                if value in _VIA_ARTICLES and value != "legacy-server":
                    kind = value
                else:
                    given = str(value)[:60]
            else:
                kind = _fallback_kind(ctx)
    except Exception:  # noqa: BLE001 — identity is never worth an exception
        pass
    return CallerVia(kind, _VIA_ARTICLES[kind], given)


def caller_fields() -> dict:
    """Client identity and access path for an event. NEVER raises.

    Unavailable identity still carries the default access kind, `dev`.
    """
    via = resolve_via()
    out: dict = {"via": via.kind}
    if via.given is not None:
        out["via_given"] = via.given
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
        pass
    return out
