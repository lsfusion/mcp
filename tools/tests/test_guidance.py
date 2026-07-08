"""Tests for tools/guidance.py — the get_guidance proxy.

`urllib.request.urlopen` is monkeypatched so no network call happens; we only
assert that each URL is fetched, decoded, and concatenated in order, and that
fetch failures propagate (no cache/fallback).
"""

from __future__ import annotations

import io
import urllib.error

import pytest

import tools.guidance as g


class _FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def test_fetches_and_concatenates_in_order(monkeypatch):
    seen: list[tuple[str, float]] = []

    def fake_urlopen(url, timeout=None):
        seen.append((url, timeout))
        return _FakeResp(f"body of {url}".encode("utf-8"))

    monkeypatch.setattr(g.urllib.request, "urlopen", fake_urlopen)

    out = g.fetch_guidance(["http://a/Brief.md", "http://b/Rules.md"], timeout=3)

    assert out == "body of http://a/Brief.md\n\nbody of http://b/Rules.md"
    assert [u for u, _ in seen] == ["http://a/Brief.md", "http://b/Rules.md"]
    assert all(t == 3 for _, t in seen)


def test_defaults_to_brief_and_rules_urls(monkeypatch):
    seen: list[str] = []

    def fake_urlopen(url, timeout=None):
        seen.append(url)
        return _FakeResp(b"x")

    monkeypatch.setattr(g.urllib.request, "urlopen", fake_urlopen)

    g.fetch_guidance()

    assert seen == [g.BRIEF_URL, g.RULES_URL]


def test_failure_propagates(monkeypatch):
    def fake_urlopen(url, timeout=None):
        raise urllib.error.URLError("boom")

    monkeypatch.setattr(g.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(urllib.error.URLError):
        g.fetch_guidance(["http://a/Brief.md"])


def test_version_is_stable_and_content_derived():
    assert g.guidance_version("same") == g.guidance_version("same")
    assert g.guidance_version("a") != g.guidance_version("b")
    # 12 hex chars, the SHA-256 prefix
    v = g.guidance_version("x")
    assert len(v) == 12 and all(c in "0123456789abcdef" for c in v)


def test_stamped_guidance_prefixes_matching_version_marker(monkeypatch):
    monkeypatch.setattr(
        g.urllib.request, "urlopen", lambda url, timeout=None: _FakeResp(b"BODY")
    )

    out = g.stamped_guidance(["http://a/Brief.md"])

    body = "BODY"
    assert out == f"<!-- lsfusion-guidance version: {g.guidance_version(body)} -->\n\n{body}"


def test_build_instructions_is_pure_intro_plus_body_no_fetch(monkeypatch):
    # build_instructions must NOT fetch — it only formats a pre-fetched body.
    def boom(*a, **k):
        raise AssertionError("build_instructions must not hit the network")

    monkeypatch.setattr(g.urllib.request, "urlopen", boom)

    out = g.build_instructions("STAMPED-BODY")

    assert out == f"{g.INSTRUCTIONS_INTRO}\n\nSTAMPED-BODY"
