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

import os
import urllib.request

BRIEF_URL = os.getenv("GUIDANCE_BRIEF_URL", "https://docs.lsfusion.org/Brief.md")
RULES_URL = os.getenv("GUIDANCE_RULES_URL", "https://docs.lsfusion.org/Rules.md")
FETCH_TIMEOUT = float(os.getenv("GUIDANCE_FETCH_TIMEOUT", "10"))


def fetch_guidance(urls: list[str] | None = None, timeout: float | None = None) -> str:
    """Fetch each URL and return the concatenated text. Exceptions propagate."""
    urls = [BRIEF_URL, RULES_URL] if urls is None else urls
    timeout = FETCH_TIMEOUT if timeout is None else timeout
    parts: list[str] = []
    for url in urls:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (https only, fixed hosts)
            parts.append(resp.read().decode("utf-8"))
    return "\n\n".join(parts)
