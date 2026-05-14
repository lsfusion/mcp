"""Tests for fill.config — DOCS_BASE_URL + build_source_url contract."""

from __future__ import annotations

import pytest

from fill.config import DOCS_BASE_URL, SOURCE_URL_VERSION, build_source_url


# ─── verified-against-real-server fixtures ─────────────────────────────────────
# These exact pairs were probed against https://docs.lsfusion.org on 2026-05-14
# (all returned 301 → with-slash → 200).

VERIFIED_URLS = [
    ("AGGR_operator",                 "https://docs.lsfusion.org/AGGR_operator"),
    ("Aggregations",                  "https://docs.lsfusion.org/Aggregations"),
    ("How-to_GROUP_SUM",              "https://docs.lsfusion.org/How-to_GROUP_SUM"),
    ("=_statement",                   "https://docs.lsfusion.org/=_statement"),
    ("IF_..._THEN_operator",          "https://docs.lsfusion.org/IF_..._THEN_operator"),
]


@pytest.mark.parametrize("slug,expected", VERIFIED_URLS)
def test_build_source_url_real_slugs(slug, expected):
    assert build_source_url(slug) == expected


def test_docs_base_url_has_trailing_slash():
    # Contract: DOCS_BASE_URL ends with `/`. build_source_url relies on this
    # so it doesn't have to fiddle with separator placement.
    assert DOCS_BASE_URL.endswith("/")


def test_docs_base_url_scheme_host_invariant():
    # Catches accidental edits like `http://` (no s) or wrong host. If we
    # intentionally repoint to a different host, update this test in the same
    # commit + bump SOURCE_URL_VERSION.
    assert DOCS_BASE_URL == "https://docs.lsfusion.org/"


def test_source_url_version_pinned():
    # Source-of-truth pin. Bumping this constant triggers a full reindex via
    # state.pipeline_versions drift detection, so an unintentional edit is
    # expensive. Update this test in the SAME commit when intentionally bumping.
    assert SOURCE_URL_VERSION == "v1"


# ─── rejection / safety ────────────────────────────────────────────────────────


def test_build_source_url_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        build_source_url("")


@pytest.mark.parametrize("slug", [
    "foo/bar",       # path traversal
    "foo\\bar",      # Windows path
    "foo?query=1",   # URL query
    "foo#fragment",  # URL fragment
    "foo%2Fbar",     # percent-encoded path separator
    "foo bar",       # whitespace
    "foo\nbar",      # newline
    "foo\x00bar",    # NUL
    "foo:bar",       # used by section_id grammar, not by slugs
    "foo;evil",      # semicolons
    "café",          # non-ASCII letter (Python's isalnum is Unicode-aware — must still reject)
    "ＦＯＯ",         # fullwidth alphanumerics
    "🚀",            # emoji
])
def test_build_source_url_rejects_disallowed_chars(slug):
    with pytest.raises(ValueError, match=r"does not match"):
        build_source_url(slug)


def test_build_source_url_length_cap():
    from fill.config import SLUG_MAX_LEN
    over = "A" * (SLUG_MAX_LEN + 1)
    with pytest.raises(ValueError, match="exceeds cap"):
        build_source_url(over)


def test_build_source_url_at_length_cap_ok():
    from fill.config import SLUG_MAX_LEN
    boundary = "A" * SLUG_MAX_LEN
    assert build_source_url(boundary) == f"https://docs.lsfusion.org/{boundary}"


@pytest.mark.parametrize("slug", [
    "AGGR_operator\n",     # trailing newline
    "AGGR_operator\r",     # trailing CR
    "AGGR_operator\r\n",   # CRLF
])
def test_build_source_url_rejects_trailing_newline(slug):
    # `re.match` with `^...$` would silently accept slugs ending in `\n` because
    # `$` matches before the final newline. `fullmatch` closes this hole.
    with pytest.raises(ValueError, match="does not match"):
        build_source_url(slug)


@pytest.mark.parametrize("slug", [".", ".."])
def test_build_source_url_rejects_dot_segments(slug):
    with pytest.raises(ValueError, match="dot-segment"):
        build_source_url(slug)


# ─── shape ─────────────────────────────────────────────────────────────────────


def test_build_source_url_no_double_slash():
    # DOCS_BASE_URL ends with `/`, slug must not start with `/`. We don't
    # add a separator, so the result has exactly one slash between base and slug.
    url = build_source_url("AGGR_operator")
    # Count slashes after the scheme.
    body = url.split("://", 1)[1]
    assert "//" not in body, f"unexpected double slash in {url}"


def test_build_source_url_idempotent():
    # Pure function — same input, same output.
    assert build_source_url("Foo") == build_source_url("Foo")
