"""fill.config — runtime constants and citation-URL helper.

Single source of truth for `DOCS_BASE_URL` and the `build_source_url(slug)`
function. Consumed by:
  - `fill.ingest` to compute the `source_url` per-section attribute on upload.
  - `mcp.tools.rag_retrieve` (indirectly, via the attribute) for citation links
    returned to the agent.

`SOURCE_URL_VERSION` participates in `section_payload_hash` (see plan §
"Chunking algorithm") so a change to `DOCS_BASE_URL` or `build_source_url`
behavior automatically forces a full reindex via Step-0 version-drift
detection. **Bump the version constant when changing the URL contract.**
"""

from __future__ import annotations

import re

# Single source of truth for slug shape: platform/docs/<lang> slugs allow
# letters/digits/_/-/=/. (no /, no whitespace, no :). ASCII-only on purpose —
# Python's str.isalnum() is Unicode-aware and would let `é` through.
SLUG_RE = re.compile(r"^[A-Za-z0-9_\-=.]+$")

# Length cap on the slug — defense-in-depth guard. 256 chars matches the budget
# carried by other identity attributes (see plan §"Attribute length validation").
SLUG_MAX_LEN = 256

# Verified against live https://docs.lsfusion.org as of 2026-05-14:
#   /AGGR_operator           → 301 → /AGGR_operator/             → 200
#   /Aggregations            → 301 → /Aggregations/              → 200
#   /How-to_GROUP_SUM        → 301 → /How-to_GROUP_SUM/          → 200
#   /=_statement             → 301 → /=_statement/               → 200
#   /IF_..._THEN_operator    → 301 → /IF_..._THEN_operator/      → 200
# Both with and without trailing slash resolve to the same content; the
# no-slash form is what we publish in attributes (one extra redirect hop
# for browsers, irrelevant for the LLM agent that consumes the URL).
DOCS_BASE_URL: str = "https://docs.lsfusion.org/"

# Bump when DOCS_BASE_URL or build_source_url logic changes — triggers full
# reindex via state.pipeline_versions drift detection (see plan).
SOURCE_URL_VERSION: str = "v1"


def build_source_url(slug: str) -> str:
    """Return the canonical page URL for a doc slug.

    Page-level only — does not include section anchors. Section identity is
    surfaced separately via the `heading_path` attribute (see plan §
    "Retrieval policy"); synthetic anchors (`task-001`, `::part-NN`) do not
    correspond to real Docusaurus URL fragments and are intentionally absent.

    Slug must be a non-empty string with no path separators, matching SLUG_RE.
    """
    if not slug:
        raise ValueError("build_source_url: slug must be non-empty")
    if len(slug) > SLUG_MAX_LEN:
        raise ValueError(
            f"build_source_url: slug length {len(slug)} exceeds cap {SLUG_MAX_LEN}"
        )
    # `fullmatch` rather than `match` — `$` would otherwise accept a trailing
    # newline because in Python regex `$` matches before the final `\n`.
    if not SLUG_RE.fullmatch(slug):
        raise ValueError(
            f"build_source_url: slug does not match {SLUG_RE.pattern!r}: {slug!r}"
        )
    # Reject dot-segments — `.` and `..` are valid per regex but clients may
    # path-normalize them away from the intended URL.
    if slug in (".", ".."):
        raise ValueError(f"build_source_url: dot-segment slug not allowed: {slug!r}")
    return f"{DOCS_BASE_URL}{slug}"
