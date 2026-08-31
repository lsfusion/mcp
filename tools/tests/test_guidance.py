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
    def __init__(self, data, url="https://docs.lsfusion.org/x.md"):
        super().__init__(data)
        self._url = url

    def geturl(self):
        # By default the response landed where it was asked for; a test that
        # simulates a redirect passes a different url.
        return self._url

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


def test_stamped_guidance_heads_body_with_marker_and_notice(monkeypatch):
    monkeypatch.setattr(
        g.urllib.request, "urlopen", lambda url, timeout=None: _FakeResp(b"BODY", url)
    )

    out = g.stamped_guidance(["http://a/Brief.md"])

    body = "BODY"
    marker = f"<!-- lsfusion-guidance version: {g.guidance_version(body)} -->"
    assert out.startswith(f"{marker}\n{g.GUIDANCE_NOTICE}\n\n")
    # Version stamps the BODY alone, so it matches an independent hash of the docs.
    assert g.guidance_version(body) in out


def test_the_start_of_session_call_fences_each_top_article(monkeypatch):
    """The call that needs the fence most: it is made on every task, it is the
    largest result this server returns, and it carries the maps — so a silent
    truncation here costs the assistant not one article but its knowledge that
    the others exist."""
    monkeypatch.setattr(
        g.urllib.request, "urlopen",
        lambda url, timeout=None: _FakeResp(f"body of {url}".encode(), url),
    )

    out = g.stamped_guidance()

    for branch, url in (("brief", g.BRIEF_URL), ("rules", g.RULES_URL)):
        body = f"body of {url}"
        assert f"=== BEGIN lsfusion {branch}/top | rev {g.guidance_version(body)} | chars {len(body)} ===" in out
        assert f"=== END lsfusion {branch}/top | chars {len(body)} ===" in out
    assert out.rstrip().endswith("===")


def test_server_instructions_fit_under_client_truncation_cap():
    # Clients cap `instructions` (Claude Code at ~2 KB). If this text ever grows
    # past the cap it arrives mutilated, so guard the budget explicitly.
    assert len(g.SERVER_INSTRUCTIONS.encode("utf-8")) < 2000


def test_server_instructions_point_to_the_tool_and_never_claim_in_context():
    instr = g.SERVER_INSTRUCTIONS
    assert "lsfusion_get_guidance" in instr
    # The regression that made this necessary: telling the assistant the rules
    # are already present, while the client had silently truncated them away.
    lowered = instr.lower()
    assert "already in context" not in lowered
    assert "do not need to call" not in lowered
    assert "not included here" in lowered


def test_notice_does_not_claim_completeness_or_flatten_rule_strength():
    # Two regressions this guards, both of which the notice used to carry:
    #   1. "the COMPLETE lsFusion brief and rules" — false once the per-area
    #      rules live in their own articles that get_guidance does not fetch.
    #   2. "the rules it omits are mandatory" / "strictly follow every rule" —
    #      the guidance contains both MUST and SHOULD, and levelling them
    #      silently promotes every SHOULD to a binding requirement.
    lowered = g.GUIDANCE_NOTICE.lower()
    assert "complete lsfusion brief and rules" not in lowered
    assert "strictly follow" not in lowered
    #   3. pointing at `retrieve_docs(type='rules')` — that branch left the
    #      search corpus; an area's rules are read whole, by name.
    assert "retrieve_docs" not in lowered
    assert "lsfusion_get_guidance(rules='<area>')" in g.GUIDANCE_NOTICE
    assert "should" in lowered and "must" in lowered


def test_server_instructions_do_not_flatten_rule_strength():
    lowered = g.SERVER_INSTRUCTIONS.lower()
    assert "strictly follow" not in lowered
    assert "stated strength" in lowered


# --- reading ONE article by name ---------------------------------------------


class _FakeHTTPError(urllib.error.HTTPError):
    def __init__(self, code):
        super().__init__("http://x", code, "err", {}, None)


def test_area_name_maps_to_the_branch_prefixed_slug():
    assert g.article_url("rules", "logic") == f"{g.GUIDANCE_BASE_URL}Rules_logic.md"
    assert g.article_url("brief", "interface") == f"{g.GUIDANCE_BASE_URL}Brief_interface.md"
    # Case and surrounding space are the caller's, not an error.
    assert g.article_url("rules", "  Logic ") == f"{g.GUIDANCE_BASE_URL}Rules_logic.md"


def test_top_aliases_reach_the_branch_top_article():
    for alias in ("top", "rules", "index", "map"):
        assert g.article_url("rules", alias) == f"{g.GUIDANCE_BASE_URL}Rules.md"
    assert g.article_url("brief", "brief") == f"{g.GUIDANCE_BASE_URL}Brief.md"


def test_a_name_cannot_escape_its_branch_or_walk_the_path():
    # The branch lives in the parameter, never in the value, and the prefix is
    # ours — so no caller-supplied name can name a page in the other branch or
    # anywhere else on the site.
    for hostile in ("../Brief", "Brief_forms", "a/b", "x.md", "", "UPPER/", "logic;rm"):
        with pytest.raises(ValueError):
            g.article_slug("rules", hostile)


def test_unresolvable_name_is_an_answer_not_an_exception_and_makes_no_request(monkeypatch):
    def fake_urlopen(url, timeout=None):
        raise AssertionError("must not reach the network for a name that cannot be a slug")

    monkeypatch.setattr(g.urllib.request, "urlopen", fake_urlopen)

    out = g.read_article("rules", "Brief_forms")

    assert "NO SUCH GUIDANCE ARTICLE" in out
    # The load-bearing sentence: an empty answer must never read as "no rules apply".
    assert "NOTHING WAS READ" in out
    assert "lsfusion_get_guidance()" in out


def test_404_is_answered_as_text_by_status_not_by_body(monkeypatch):
    # The docs site serves a full HTML 404 page, so the body is a poor signal
    # and a long one; only the status tells us the article does not exist.
    def fake_urlopen(url, timeout=None):
        raise _FakeHTTPError(404)

    monkeypatch.setattr(g.urllib.request, "urlopen", fake_urlopen)

    out = g.read_article("rules", "nosucharea")
    assert "NO SUCH GUIDANCE ARTICLE" in out and "NOTHING WAS READ" in out


def test_other_failures_propagate_and_are_never_dressed_up_as_absence(monkeypatch):
    for boom in (_FakeHTTPError(500), urllib.error.URLError("down"), TimeoutError()):
        monkeypatch.setattr(
            g.urllib.request, "urlopen", lambda url, timeout=None, e=boom: (_ for _ in ()).throw(e)
        )
        with pytest.raises(type(boom)):
            g.read_article("rules", "logic")


def test_article_is_fenced_and_the_end_fence_terminates_the_result(monkeypatch):
    body = "RULE BODY"
    monkeypatch.setattr(
        g.urllib.request, "urlopen", lambda url, timeout=None: _FakeResp(body.encode(), url)
    )

    out = g.read_article("rules", "logic")

    # Completeness is proved by a TERMINATOR, not a flag: a header claiming
    # completeness survives truncation and then lies, an END fence cannot.
    assert out.endswith(f"=== END lsfusion rules/logic | chars {len(body)} ===")
    assert f"rev {g.guidance_version(body)}" in out
    assert f"| chars {len(body)} ===\n{body}\n" in out
    # The version stamps the BODY alone, so it equals a hash of the published page.
    assert g.guidance_version(body) == g.guidance_version(body)


def test_unknown_branch_is_a_programming_error():
    with pytest.raises(ValueError):
        g.article_slug("howto", "logic")


def test_a_redirect_is_not_the_article_that_was_asked_for(monkeypatch):
    """Sanitizing the name secures the URL we ASK for, not the page we get.
    urlopen follows redirects silently, so without this a moved slug could hand
    back a different article — the other branch's, even — framed as complete."""
    monkeypatch.setattr(
        g.urllib.request, "urlopen",
        lambda url, timeout=None: _FakeResp(b"someone else's article",
                                            "https://docs.lsfusion.org/Brief_logic.md"))

    out = g.read_article("rules", "logic")
    assert "NO SUCH GUIDANCE ARTICLE" in out and "redirected" in out
    assert "someone else's article" not in out


def test_a_soft_404_page_is_not_served_as_an_article(monkeypatch):
    monkeypatch.setattr(g.urllib.request, "urlopen",
                        lambda url, timeout=None: _FakeResp(b"<!doctype html><h1>Not found</h1>", url))
    out = g.read_article("rules", "logic")
    assert "NO SUCH GUIDANCE ARTICLE" in out and "not the article" in out
