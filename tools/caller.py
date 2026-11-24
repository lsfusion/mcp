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
because it answers the question that prompted it. For the IntelliJ plugin the
User-Agent also names the IDE build, kept as its own `ide_build` field.

`x-agent-model` is read if a caller sets it: nothing sends it today, and Claude
Code's own header config is reported not to reach `tools/call` at all, but a
harness comparing models can set it on a proxy, and reading it costs one dict
lookup. Absent it, the field is simply missing rather than guessed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_MODEL_HEADER = "x-agent-model"

# The IntelliJ plugin sends `lsfusion-intellij/<plugin version> <IDE build>`,
# the build as IntelliJ's BuildNumber.asString() writes it: an optional product
# code, then the number (`IU-253.28294.334`, `AI-243.22562.218.2431.12345`,
# `IU-253.SNAPSHOT`). Only that shape yields an `ide_build`.
_PLUGIN_CLIENT = "lsfusion-intellij"
_PLUGIN_USER_AGENT = re.compile(
    r"lsfusion-intellij/\S+ ((?:[A-Z]{1,10}-)?\d+(?:\.(?:\d+|SNAPSHOT))*)")

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


def _ide_build(client, user_agent) -> str | None:
    """The IDE build the IntelliJ plugin names in its User-Agent, else None.

    The same plugin version behaves differently on different IDE builds, and
    clientInfo carries only the plugin version. Taken only when the handshake
    named the plugin and the header has the plugin's exact shape; another
    client's User-Agent yields nothing here.
    """
    if client != _PLUGIN_CLIENT or not user_agent:
        return None
    m = _PLUGIN_USER_AGENT.fullmatch(str(user_agent).strip())
    return m.group(1)[:40] if m else None


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
            build = _ide_build(out.get("client"), ua)  # from the whole header, not the capped one
            if build:
                out["ide_build"] = build
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
