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

# The MCP `instructions` field, returned at the `initialize` handshake.
#
# It carries only a POINTER, never the guidance itself. Clients truncate this
# field aggressively (Claude Code cuts it at ~2 KB), so a ~50 KB body would
# arrive mutilated with no way to recover the tail — and, worse, any claim that
# "the rules are already in context" would then be a lie the assistant acts on.
# Keep this text short enough to survive any client's cap intact.
#
# Two things this text must NOT do, both of which it used to do: claim the
# returned set is complete (the per-area rules articles are not in it), and
# flatten rule strength ("strictly follow every rule") — the guidance contains
# both MUST and SHOULD, and levelling them silently promotes every SHOULD.
SERVER_INSTRUCTIONS = (
    "The lsFusion guidance — a brief capability overview plus the core coding "
    "rules — is NOT included here. It is large (tens of kilobytes) and is "
    "served by the `lsfusion_get_guidance` tool.\n\n"
    "Before starting ANY lsFusion task — writing, modifying or reviewing "
    "lsFusion code (`.lsf`), designing forms, properties or actions, or "
    "answering questions about lsFusion syntax or semantics — call "
    "`lsfusion_get_guidance` and apply every rule it returns according to that "
    "rule's stated strength (MUST / MUST NOT are binding; SHOULD / SHOULD NOT "
    "are recommendations). Once per session is enough. If your client saves "
    "the result to a file and shows only a preview, read that file in full "
    "before continuing.\n\n"
    "What it returns is the TOP LEVEL only: the rules for a specific area are "
    "separate articles, retrieved with `lsfusion_retrieve_docs(type='rules')`.\n\n"
    "Other tools: `lsfusion_retrieve_docs` searches the official documentation; "
    "`lsfusion_report_feedback` submits a consented, depersonalized quality "
    "signal (the guidance says when)."
)

# First lines of every `lsfusion_get_guidance` result. A client that persists an
# oversized tool result to a file still shows the assistant a preview of the
# head, so this notice is the one part guaranteed to be read — spend it telling
# the assistant how to recover the part it cannot see, and where the rules that
# are NOT in this response live.
GUIDANCE_NOTICE = (
    "> IMPORTANT — this message contains the top-level lsFusion brief and the "
    "CORE rules; it is not every rule that applies. The rules for a specific "
    "area are separate articles: retrieve them with "
    "`lsfusion_retrieve_docs(type='rules', query='<area>')` before working in "
    "that area. If your client truncated this message, or saved it to a file "
    "and showed you only a preview, read that file IN FULL before writing or "
    "reviewing any lsFusion code. Apply each rule at its stated strength: "
    "MUST / MUST NOT are binding, SHOULD / SHOULD NOT are recommendations."
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
    cheap to compare, so a client (or a reviewer) can tell which revision of the
    published guidance a session actually received.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _version_marker(text: str) -> str:
    return f"<!-- lsfusion-guidance version: {guidance_version(text)} -->"


def stamped_guidance(urls: list[str] | None = None, timeout: float | None = None) -> str:
    """Live guidance body, headed by its version marker and the recovery notice.

    The version stamps the BODY alone, so it equals an independent hash of the
    published Brief+Rules and stays stable as the notice wording evolves.
    """
    body = fetch_guidance(urls, timeout)
    return f"{_version_marker(body)}\n{GUIDANCE_NOTICE}\n\n{body}"
