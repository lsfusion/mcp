"""get_guidance source — fetch the published Brief/Rules Markdown.

A thin proxy over the docs site's per-page Markdown twins, emitted by
`@signalwire/docusaurus-plugin-llms-txt` at `/<slug>.md` (e.g.
https://docs.lsfusion.org/Brief.md). The published site is the single source
of truth: no cache, no packaged fallback. If a page is unreachable the call
fails (the timeout prevents a hang) rather than returning stale guidance.

Kept free of FastMCP/SDK imports so it is unit-testable on its own; `server.py`
just wires `lsfusion_get_guidance` to `fetch_guidance()`.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request

BRIEF_URL = os.getenv("GUIDANCE_BRIEF_URL", "https://docs.lsfusion.org/Brief.md")
RULES_URL = os.getenv("GUIDANCE_RULES_URL", "https://docs.lsfusion.org/Rules.md")
FETCH_TIMEOUT = float(os.getenv("GUIDANCE_FETCH_TIMEOUT", "10"))

# Prepended to the auto-delivered handshake instructions (MCP `instructions`
# field). Tells the assistant the rules are already in context, so it need not
# spend a tool call on `lsfusion_get_guidance` unless this block is absent.
INSTRUCTIONS_INTRO = (
    "The brief overview and mandatory rules for working with lsFusion are "
    "delivered automatically below at connection time. Read them and strictly "
    "follow every rule. Because they are already in context, you do NOT need to "
    "call `lsfusion_get_guidance` — call it only if this block is missing (e.g. "
    "after context compaction dropped it)."
)


def fetch_guidance(urls: list[str] | None = None, timeout: float | None = None) -> str:
    """Fetch each URL and return the concatenated text. Exceptions propagate."""
    urls = [BRIEF_URL, RULES_URL] if urls is None else urls
    timeout = FETCH_TIMEOUT if timeout is None else timeout
    parts: list[str] = []
    for url in urls:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (https only, fixed hosts)
            parts.append(resp.read().decode("utf-8"))
    return "\n\n".join(parts)


def guidance_version(text: str) -> str:
    """Short, content-derived version stamp for a guidance body.

    A truncated SHA-256 of the exact text. Stable for identical content and
    cheap to compare, so a client (e.g. a SessionStart hook) can detect when a
    cached local copy has drifted from what the server now serves.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _version_marker(text: str) -> str:
    return f"<!-- lsfusion-guidance version: {guidance_version(text)} -->"


def stamped_guidance(urls: list[str] | None = None, timeout: float | None = None) -> str:
    """Live guidance body prefixed with its version marker.

    Both delivery channels — the `lsfusion_get_guidance` tool result and the
    handshake `instructions` — emit the SAME marker for the same content, so the
    version a session sees is unambiguous regardless of how it arrived.
    """
    body = fetch_guidance(urls, timeout)
    return f"{_version_marker(body)}\n\n{body}"


def build_instructions(stamped_body: str) -> str:
    """Wrap an already-fetched stamped guidance body as the MCP `instructions` payload.

    Pure formatter (no network): the caller fetches once via `stamped_guidance()`
    and feeds the result here AND to the `lsfusion_get_guidance` tool, so both
    delivery channels serve byte-identical content and version marker.
    """
    return f"{INSTRUCTIONS_INTRO}\n\n{stamped_body}"
