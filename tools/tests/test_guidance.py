"""Tests for tools/guidance.py — the get_guidance proxy.

`urllib.request.urlopen` is monkeypatched so no network call happens; we only
assert that each URL is fetched, decoded, and concatenated in order, and that
fetch failures propagate (no cache/fallback).
"""

from __future__ import annotations

import io
import re
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


@pytest.mark.parametrize("via,client,kind,names", [
    ("dev", "Claude Code", "dev", ["via_dev"]),
    ("idedev", "Claude Code", "idedev", ["via_dev", "via_idedev"]),
    ("app", "lsfusion-server", "app", ["via_app"]),
    ("webapp", "lsfusion-server", "webapp", ["via_app", "via_webapp"]),
    (None, "lsfusion-server", "legacy-server", ["via_app", "via_webapp"]),
    (None, "lsfusion-intellij", "idedev", ["via_dev", "via_idedev"]),
    (None, "Claude Code", "dev", ["via_dev"]),
    ("unknown", "lsfusion-server", "dev", ["via_dev"]),
])
def test_start_answer_assembly_matrix(monkeypatch, caller_context, via, client, kind, names):
    caller_context(via=via, name=client)
    seen = []

    def fetch(url, timeout=None):
        seen.append((url, timeout))
        return _FakeResp(f"body of {url}".encode(), url)

    monkeypatch.setattr(g.urllib.request, "urlopen", fetch)
    calls = _capture(monkeypatch)

    out = g.stamped_guidance(timeout=3)

    urls = ([g.BRIEF_URL] + [g.article_url("brief", name) for name in names]
            + [g.RULES_URL] + [g.article_url("rules", name) for name in names])
    labels = (["brief/top"] + [f"brief/{name}" for name in names]
              + ["rules/top"] + [f"rules/{name}" for name in names])
    assert seen == [(url, 3) for url in urls]
    assert re.findall(r"^=== BEGIN lsfusion (\S+) \|", out, re.M) == labels
    assert re.findall(r"^=== END lsfusion (\S+) \|", out, re.M) == labels
    bodies = [f"body of {url}" for url in urls]
    body = "\n\n".join(bodies)
    blocks = [g.fenced(*label.split("/"), text) for label, text in zip(labels, bodies)]
    assert out == f"<!-- lsfusion-guidance version: {g.guidance_version(body)} -->\n{g.GUIDANCE_NOTICE}\n\n" + "\n\n".join(blocks)
    (event,) = calls
    assert event["ok"] is True
    assert event["fields"]["via"] == kind
    assert event["fields"]["via_articles"] == names
    assert event["fields"]["rev"] == g.guidance_version(body)
    assert event["fields"]["chars"] == len(body)


def test_explicit_urls_do_not_add_path_articles(monkeypatch, caller_context):
    caller_context(via="webapp")
    seen = []

    def fetch(url, timeout=None):
        seen.append(url)
        return _FakeResp(b"BODY", url)

    monkeypatch.setattr(g.urllib.request, "urlopen", fetch)
    calls = _capture(monkeypatch)
    out = g.stamped_guidance(["http://a/Brief.md"])
    assert seen == ["http://a/Brief.md"]
    assert out.endswith(g.fenced("brief", "top", "BODY"))
    assert "=== BEGIN lsfusion rules/" not in out
    assert calls[0]["fields"]["via_articles"] == []


@pytest.mark.parametrize("via,client,name", [
    ("dev", "Claude Code", "via_dev"),
    ("idedev", "Claude Code", "via_dev"),
    ("idedev", "Claude Code", "via_idedev"),
    ("app", "lsfusion-server", "via_app"),
    ("webapp", "lsfusion-server", "via_app"),
    ("webapp", "lsfusion-server", "via_webapp"),
    (None, "lsfusion-server", "via_webapp"),
])
@pytest.mark.parametrize("failure", ["404", "500", "network", "timeout", "redirect", "html"])
@pytest.mark.parametrize("branch", ["brief", "rules"])
def test_any_failed_via_article_fails_the_whole_top_call(
        monkeypatch, caller_context, via, client, name, failure, branch):
    caller_context(via=via, name=client)
    target = g.article_url(branch, name)
    error_type = {
        "404": g._NotThisArticle,
        "500": _FakeHTTPError,
        "network": urllib.error.URLError,
        "timeout": TimeoutError,
        "redirect": g._NotThisArticle,
        "html": g._NotThisArticle,
    }[failure]

    def fetch(url, timeout=None):
        if url != target:
            return _FakeResp(b"VALID BODY", url)
        if failure in ("404", "500"):
            raise _FakeHTTPError(int(failure))
        if failure == "network":
            raise urllib.error.URLError("down")
        if failure == "timeout":
            raise TimeoutError()
        if failure == "redirect":
            return _FakeResp(b"OTHER ARTICLE", g.BRIEF_URL)
        return _FakeResp(b" \n<!doctype html><h1>Not found</h1>", url)

    monkeypatch.setattr(g.urllib.request, "urlopen", fetch)
    calls = _capture(monkeypatch)
    with pytest.raises(error_type):
        g.stamped_guidance()
    (event,) = calls
    assert event["ok"] is False
    assert event["fields"]["branch"] == "top"
    assert event["fields"]["outcome"] == "error"
    assert event["fields"]["error"] == error_type.__name__
    assert event["fields"]["via_articles"] == []  # No partial answer was served.


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
    assert "access kind" in lowered and "table of access kinds" in lowered


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
    assert g.article_url("rules", "via_idedev") == f"{g.GUIDANCE_BASE_URL}Rules_via_idedev.md"


def test_named_path_article_is_read_alone_regardless_of_current_path(monkeypatch, caller_context):
    caller_context(via="app")
    seen = []

    def fetch(url, timeout=None):
        seen.append(url)
        return _FakeResp(b"IDE RULES", url)

    monkeypatch.setattr(g.urllib.request, "urlopen", fetch)
    out = g.read_article("rules", "via_idedev")
    assert seen == [g.article_url("rules", "via_idedev")]
    assert out.endswith(g.fenced("rules", "via_idedev", "IDE RULES"))
    assert "rules/via_app" not in out and "rules/via_dev" not in out


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


# --- every call is logged, and the outcome is classified --------------------


def _capture(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(g, "emit", lambda event, fields, *, stream, ok=True:
                        calls.append({"event": event, "fields": fields, "stream": stream, "ok": ok}))
    return calls


def test_a_served_article_logs_ok_with_its_size_and_revision(monkeypatch):
    body = "RULE BODY"
    monkeypatch.setattr(g.urllib.request, "urlopen", lambda url, timeout=None: _FakeResp(body.encode(), url))
    calls = _capture(monkeypatch)

    g.read_article("rules", "logic")

    (c,) = calls
    assert c["event"] == "get_guidance" and c["stream"] == "retrieval" and c["ok"] is True
    f = c["fields"]
    assert f["branch"] == "rules" and f["area"] == "logic" and f["outcome"] == "ok"
    assert f["chars"] == len(body) and f["rev"] == g.guidance_version(body)
    assert "text" not in f and body not in str(f)


def test_a_miss_is_logged_as_not_found_not_as_an_error(monkeypatch):
    # A bad name and a 404 are ANSWERS to the caller; the log must say so too,
    # or every typo would read as an outage in the adoption numbers.
    calls = _capture(monkeypatch)
    monkeypatch.setattr(g.urllib.request, "urlopen",
                        lambda url, timeout=None: (_ for _ in ()).throw(_FakeHTTPError(404)))
    g.read_article("rules", "nosuch")
    g.read_article("rules", "Brief_forms")  # rejected before any request
    assert [c["fields"]["outcome"] for c in calls] == ["not_found", "not_found"]
    assert all(c["ok"] is False for c in calls)


def test_a_failure_is_logged_as_error_and_still_propagates(monkeypatch):
    calls = _capture(monkeypatch)
    monkeypatch.setattr(g.urllib.request, "urlopen",
                        lambda url, timeout=None: (_ for _ in ()).throw(urllib.error.URLError("down")))
    with pytest.raises(urllib.error.URLError):
        g.read_article("brief", "view")
    (c,) = calls
    assert c["fields"]["outcome"] == "error" and c["ok"] is False
    assert "URLError" in c["fields"]["error"]


def test_the_start_of_session_call_is_logged_as_top(monkeypatch):
    monkeypatch.setattr(g.urllib.request, "urlopen",
                        lambda url, timeout=None: _FakeResp(f"body of {url}".encode(), url))
    calls = _capture(monkeypatch)
    g.stamped_guidance()
    (c,) = calls
    assert c["fields"]["branch"] == "top" and c["fields"]["area"] is None
    assert c["fields"]["outcome"] == "ok"
    assert c["fields"]["via"] == "dev"
    assert c["fields"]["via_articles"] == ["via_dev"]


def test_logging_can_never_break_the_call(monkeypatch):
    monkeypatch.setattr(g.urllib.request, "urlopen", lambda url, timeout=None: _FakeResp(b"x", url))
    monkeypatch.setattr(g, "emit", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")))
    assert "=== END" in g.read_article("rules", "logic")
