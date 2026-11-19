"""Tests for tools/feedback_issues.py — rendering, neutralizing, labels, the
inbox -> processed move, crash recovery and rate limits, against a fake GitHub."""
from __future__ import annotations

import io
import json
import pathlib
import re
import urllib.parse

import pytest

import tools.feedback_issues as fi
from tools.feedback_issues import API, AREAS, BODY_MAX, LABELS, ZWSP

REPO = "lsfusion/platform-feedback"


# --- helpers -------------------------------------------------------------------

def _record(rid: str = "rpt_00000000000000a1", **over) -> dict:
    rec = {
        "report_id": rid,
        "ts": "2026-11-19T08:42:49.356Z",
        "schema_version": 5,
        "server_version": "64e52fa40f057ef786147b0a25f60b1082ad5322",
        "client": "claude-code",
        "client_version": "2.1.327",
        "dedup_fingerprint": "1651cf82a25c17ec",
        "agent_journey_id": "j-1",
        "signal_type": "doc-gap",
        "problem_summary": "group sum by customer",
        "recommendation": {"primary_target": "how-to", "secondary_targets": [],
                           "suggested_change": "add a GROUP SUM example", "confidence": "medium"},
        "eval_errors": [],
        "retrieve_queries": [],
        "retrieved_docs_summary": [],
        "final_outcome": "fixed",
        "lsfusion_version": "6.1",
        "agent": "claude-code, claude-opus-4-5",
    }
    rec.update(over)
    return rec


def _name(i: int, rid: str) -> str:
    return f"20261119T08{i:02d}00-{rid}.json"


@pytest.fixture
def reports(tmp_path) -> pathlib.Path:
    (tmp_path / "reports" / "inbox").mkdir(parents=True)
    return tmp_path / "reports"


def _put(reports: pathlib.Path, i: int, rec: dict) -> str:
    name = _name(i, rec["report_id"])
    (reports / "inbox" / name).write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return name


def _inbox(reports: pathlib.Path) -> list[str]:
    return sorted(p.name for p in (reports / "inbox").iterdir())


def _processed(reports: pathlib.Path, name: str) -> dict:
    return json.loads((reports / "processed" / name).read_text(encoding="utf-8"))


class FakeGitHub:
    """Labels and issues of one repo, served over the transport interface."""

    def __init__(self, labels=("bug", "documentation", "good first issue")):
        self.labels = [{"name": n} for n in labels]
        self.issues: list[dict] = []
        self.calls: list[tuple] = []
        self.fail: dict[tuple, list] = {}  # (method, path) -> queued (status, headers, payload)
        self.page_size = 100

    def add_issue(self, body: str, created_at: str, updated_at: str | None = None) -> dict:
        number = len(self.issues) + 1
        issue = {"number": number, "body": body, "created_at": created_at,
                 "updated_at": updated_at or created_at,
                 "html_url": f"https://github.com/{REPO}/issues/{number}"}
        self.issues.append(issue)
        return issue

    def posted(self) -> list[dict]:
        return [c[2] for c in self.calls if c[:2] == ("POST", f"/repos/{REPO}/issues")]

    def __call__(self, method, url, headers, data):
        assert url.startswith(API + "/")
        parsed = urllib.parse.urlsplit(url[len(API):])
        path, query = parsed.path, dict(urllib.parse.parse_qsl(parsed.query))
        body = json.loads(data) if data else None
        self.calls.append((method, path, body, headers))
        queued = self.fail.get((method, path))
        if queued:
            status, resp_headers, payload = queued.pop(0)
            return status, resp_headers, json.dumps(payload).encode()
        if (method, path) == ("GET", f"/repos/{REPO}/labels"):
            return 200, {}, json.dumps(self.labels).encode()
        if (method, path) == ("POST", f"/repos/{REPO}/labels"):
            self.labels.append(body)
            return 201, {}, json.dumps(body).encode()
        if (method, path) == ("GET", f"/repos/{REPO}/issues"):
            # As GitHub does: `since` filters on updated_at, the order is by created_at.
            items = [i for i in self.issues if i["updated_at"] >= query.get("since", "")]
            items.sort(key=lambda i: i["created_at"], reverse=query.get("direction") == "desc")
            page = int(query.get("page", "1"))
            chunk = items[(page - 1) * self.page_size: page * self.page_size]
            headers_out = {}
            if page * self.page_size < len(items):
                nxt = dict(query, page=str(page + 1))
                headers_out["Link"] = f'<{API}{path}?{urllib.parse.urlencode(nxt)}>; rel="next"'
            return 200, headers_out, json.dumps(chunk).encode()
        if (method, path) == ("POST", f"/repos/{REPO}/issues"):
            issue = self.add_issue(body["body"], f"2026-11-19T12:00:{len(self.issues):02d}Z")
            return 201, {}, json.dumps({"number": issue["number"], "html_url": issue["html_url"]}).encode()
        raise AssertionError(f"unexpected call {method} {path}")


def _main(reports, gh, *, token="ghp_testtoken", sleeps=None, argv=()):
    return fi.main(["--reports-dir", str(reports), *argv], stdin=io.StringIO(token),
                   transport=gh, sleep=(sleeps.append if sleeps is not None else lambda s: None),
                   clock=lambda: 0.0)


# --- title ---------------------------------------------------------------------

def test_title_is_a_short_summary_as_written():
    assert fi.render_title(_record()) == "group sum by customer"


def test_title_cuts_a_long_summary_at_a_word():
    words = "word " * 40
    title = fi.render_title(_record(problem_summary="  Long\n" + words))
    assert title.endswith("…") and len(title) <= fi.TITLE_MAX + 1
    assert title[:-1].endswith("word") and "\n" not in title and "  " not in title


def test_title_breaks_mentions_and_references():
    title = fi.render_title(_record(problem_summary="@someone see #12 and https://x.org"))
    assert title == f"@{ZWSP}someone see #{ZWSP}12 and https:{ZWSP}//x.org"


# --- body ----------------------------------------------------------------------

def test_body_has_the_sections_in_order_and_ends_with_the_marker():
    rec = _record(
        expectation={"expected": "a total per customer", "actual": "one total"},
        recommendation={"primary_target": "language", "secondary_targets": ["how-to", "rules"],
                        "suggested_change": "explain GROUP SUM BY", "confidence": "high",
                        "rationale": "agents miss BY"},
        eval_errors=[{"message": "Unknown property foo", "phase": "semantic",
                      "code_excerpt": "x = GROUP SUM foo(o);"}],
        final_outcome="workaround")
    body = fi.render_body(rec, _name(1, rec["report_id"]))
    order = ["### Problem", "group sum by customer", "### Expected vs actual",
             "**Expected:** a total per customer", "**Actual:** one total",
             "### Suggested change", "explain GROUP SUM BY", "**Why:** agents miss BY",
             "**Confidence:** high · **Also:** how-to, rules",
             "### Eval errors", "**semantic** · Unknown property foo",
             "```\nx = GROUP SUM foo(o);\n```",
             "**Outcome:** workaround · **Signal:** doc-gap",
             "Reported by claude-code, claude-opus-4-5 via claude-code 2.1.327 · lsFusion 6.1"
             " · MCP server `64e52fa`",
             "<details>", "</details>"]
    positions = [body.index(part) for part in order]
    assert positions == sorted(positions)
    assert body.rstrip().splitlines()[-1] == f"<!-- report_id: {rec['report_id']} -->"


def test_body_leaves_out_what_the_report_lacks():
    rec = _record()
    for key in ("client", "client_version", "lsfusion_version", "agent"):
        rec.pop(key)
    body = fi.render_body(rec, _name(1, rec["report_id"]))
    assert "### Expected vs actual" not in body and "### Eval errors" not in body
    assert "**Why:**" not in body and "**Also:**" not in body
    assert "Reported" not in body and "lsFusion" not in body
    assert "MCP server `64e52fa`" in body


def test_json_block_holds_the_record_verbatim():
    rec = _record(problem_summary="has ``` inside and @user")
    body = fi.render_body(rec, _name(1, rec["report_id"]))
    m = re.search(r"\n(`{4,})json\n(.*)\n\1\n", body, re.S)
    assert m, body
    assert json.loads(m.group(2)) == rec


def test_code_excerpt_keeps_its_text_and_outlasts_its_backticks():
    rec = _record(eval_errors=[{"message": "m", "phase": "syntax",
                                "code_excerpt": "  @defineX(a);\n```\nb <- c;"}])
    body = fi.render_body(rec, _name(1, rec["report_id"]))
    assert "````\n  @defineX(a);\n```\nb <- c;\n````" in body


def test_only_ten_eval_errors_are_shown():
    errors = [{"message": f"error {i}", "phase": "runtime"} for i in range(12)]
    body = fi.render_body(_record(eval_errors=errors), "n")
    assert "error 9" in body.split("<details>")[0] and "error 10" not in body.split("<details>")[0]
    assert "…and 2 more in the JSON below." in body


# --- neutralizing --------------------------------------------------------------

def _prose(body: str) -> str:
    """The body without its JSON block."""
    return body.split("<details>")[0]


def test_mentions_references_and_urls_are_broken():
    rec = _record(problem_summary="via a metacode (@applicationForm(options)) per lsfusion/platform "
                                  "issue #1861, GH-7, user@example.org, https://github.com/x/y/issues/2 "
                                  "and www.example.org")
    prose = _prose(fi.render_body(rec, "n"))
    assert f"@{ZWSP}applicationForm" in prose and f"#{ZWSP}1861" in prose
    assert f"GH-{ZWSP}7" in prose and f"user@{ZWSP}example.org" in prose
    assert f"https:{ZWSP}//github.com" in prose and f"www{ZWSP}.example.org" in prose
    assert not re.search(r"@\w|#\d|GH-\d|://|www\.", prose)


def test_html_is_escaped_outside_code_only():
    out = fi._md("at <module>:<line> & <b>bold</b> but `<List group=\"e\"/>` stays")
    assert out == ("at &lt;module&gt;:&lt;line&gt; &amp; &lt;b&gt;bold&lt;/b&gt; "
                   "but `<List group=\"e\"/>` stays")


def test_code_fences_in_agent_text_stay_literal_and_an_open_one_is_closed():
    assert fi._md("before <x>\n```\na <- b;\n```\nafter <y>") == \
        "before &lt;x&gt;\n```\na <- b;\n```\nafter &lt;y&gt;"
    assert fi._md("```lsf\nunclosed <z>") == "```lsf\nunclosed <z>\n```"


def test_a_fence_after_a_label_is_not_a_fence():
    # "**Why:** ```" is no fence for GitHub, so its "closing" line must not be one either.
    assert fi._md("**Why:** ```\n<a>") == "**Why:** ```\n&lt;a&gt;"


def test_links_images_and_references_are_broken():
    out = fi._md("[x](y) ![i](z) [a][b]\n[b]: http://evil")
    assert "](" not in out and "][" not in out and "]:" not in out
    assert out.count(ZWSP) == 5


def test_entities_and_escapes_cannot_smuggle_a_ping():
    assert fi._md("&#64;user &commat;user") == f"&amp;#{ZWSP}64;user &amp;commat;user"
    assert fi._md("\\@user") == f"\\@{ZWSP}user"


def test_a_table_row_is_escaped_whole():
    # In a table GitHub splits cells before it sees code spans.
    assert fi._md("| `<a>` | b |") == "| `&lt;a&gt;` | b |"


def test_an_escaped_backtick_opens_no_code_span():
    assert fi._md("\\`<b>`") == "\\`&lt;b&gt;`"


# --- size ----------------------------------------------------------------------

def test_a_long_report_fits_by_cutting_the_json_first():
    errors = [{"message": "m" * 3900, "phase": "runtime", "code_excerpt": "c" * 1900} for i in range(6)]
    rec = _record(eval_errors=errors, problem_summary="p" * 1900)
    body = fi.render_body(rec, _name(1, rec["report_id"]))
    assert len(body) <= BODY_MAX
    assert "p" * 1900 in body and _prose(body).count("m" * 3900) == 6  # sections whole
    assert "Report (JSON, cut)" in body and "processed/20261119T080100-rpt_00000000000000a1.json" in body
    assert body.rstrip().endswith(f"<!-- report_id: {rec['report_id']} -->")


def test_even_sections_too_long_for_github_fit():
    errors = [{"message": "<" * 3900, "phase": "runtime", "code_excerpt": "`" * 1900} for i in range(10)]
    rec = _record(eval_errors=errors, problem_summary="&" * 1900)
    body = fi.render_body(rec, "n")
    assert len(body) <= BODY_MAX
    assert body.rstrip().endswith(f"<!-- report_id: {rec['report_id']} -->")


# --- labels --------------------------------------------------------------------

@pytest.mark.parametrize("target", sorted(AREAS))
def test_each_issue_gets_an_area_and_its_target(target):
    labels = fi.labels_for(_record(recommendation={"primary_target": target, "suggested_change": "s"}))
    assert labels == [f"area:{AREAS[target]}", f"target:{target}"]
    assert set(labels) <= set(LABELS)


def test_the_eleven_labels_each_have_a_short_description():
    expected = {f"area:{a}" for a in AREAS.values()} | {f"target:{t}" for t in AREAS}
    assert set(LABELS) == expected and len(LABELS) == 11
    for color, description in LABELS.values():
        assert re.fullmatch(r"[0-9a-f]{6}", color) and 0 < len(description) <= 100


def test_missing_labels_are_created_and_existing_ones_untouched(reports):
    gh = FakeGitHub(labels=("bug", "documentation", "AREA:DOCS"))
    _put(reports, 1, _record())
    assert _main(reports, gh) == 0
    created = [c[2] for c in gh.calls if c[:2] == ("POST", f"/repos/{REPO}/labels")]
    assert {c["name"] for c in created} == set(LABELS) - {"area:docs"}
    assert all(c["description"] == LABELS[c["name"]][1] for c in created)
    assert {c[0] for c in gh.calls} <= {"GET", "POST"}  # nothing edited or deleted


# --- the run -------------------------------------------------------------------

def test_posts_oldest_first_and_files_each_report_as_processed(reports):
    gh = FakeGitHub()
    second = _put(reports, 2, _record("rpt_00000000000000b2", problem_summary="second"))
    first = _put(reports, 1, _record("rpt_00000000000000a1", problem_summary="first",
                                     recommendation={"primary_target": "code-bug", "suggested_change": "s"}))
    assert _main(reports, gh) == 0
    posted = gh.posted()
    assert [p["title"] for p in posted] == ["first", "second"]
    assert posted[0]["labels"] == ["area:platform", "target:code-bug"]
    assert _inbox(reports) == []
    rec = _processed(reports, first)
    assert rec["issue_number"] == 1 and rec["issue_url"].endswith("/issues/1")
    assert rec["problem_summary"] == "first" and _processed(reports, second)["issue_number"] == 2
    assert not [p for p in (reports / "processed").iterdir() if p.name.startswith(".")]


def test_temp_files_and_odd_files_are_left_alone(reports):
    gh = FakeGitHub()
    _put(reports, 1, _record())
    (reports / "inbox" / ".20261119T090000-rpt_00000000000000c3.json.tmp").write_text("{", encoding="utf-8")
    (reports / "inbox" / "notes.txt").write_text("x", encoding="utf-8")
    assert _main(reports, gh) == 0
    assert len(gh.posted()) == 1
    assert _inbox(reports) == [".20261119T090000-rpt_00000000000000c3.json.tmp", "notes.txt"]


def test_a_bad_file_fails_alone_and_stays(reports, capsys):
    gh = FakeGitHub()
    (reports / "inbox" / "20261119T080000-rpt_00000000000000x0.json").write_text("{not json", encoding="utf-8")
    bad_target = _put(reports, 1, _record("rpt_00000000000000x1",
                                          recommendation={"primary_target": "docs", "suggested_change": "s"}))
    _put(reports, 2, _record("rpt_00000000000000b2"))
    assert _main(reports, gh) == 1
    assert len(gh.posted()) == 1
    assert _inbox(reports) == ["20261119T080000-rpt_00000000000000x0.json", bad_target]
    out = capsys.readouterr().out
    assert "unknown primary_target 'docs'" in out
    assert out.strip().splitlines()[-1] == "posted 1, skipped 0, failed 2, left 0"


def test_a_crash_after_posting_does_not_post_twice(reports, monkeypatch):
    gh = FakeGitHub()
    name = _put(reports, 1, _record())
    real_write = fi._write_json

    def crash(path, record):
        raise OSError("disk full")

    monkeypatch.setattr(fi, "_write_json", crash)
    assert _main(reports, gh) == 1  # posted, but the file could not move
    assert len(gh.issues) == 1 and _inbox(reports) == [name]

    monkeypatch.setattr(fi, "_write_json", real_write)
    assert _main(reports, gh) == 0
    assert len(gh.issues) == 1  # found by its marker, not posted again
    assert _inbox(reports) == [] and _processed(reports, name)["issue_number"] == 1


def test_a_report_already_in_processed_is_only_removed(reports):
    gh = FakeGitHub()
    name = _put(reports, 1, _record())
    (reports / "processed").mkdir()
    (reports / "processed" / name).write_text(json.dumps({**_record(), "issue_number": 7, "issue_url": "u"}),
                                              encoding="utf-8")
    assert _main(reports, gh) == 0
    assert gh.posted() == [] and _inbox(reports) == []
    assert _processed(reports, name)["issue_number"] == 7


def test_the_scan_pages_through_issues_and_stops_at_the_oldest_report(reports):
    gh = FakeGitHub()
    gh.page_size = 10
    marked = gh.add_issue("x\n\n<!-- report_id: rpt_00000000000000a1 -->\n", "2026-11-19T09:00:00Z")
    for i in range(25):  # newer issues push the marked one to the third page
        gh.add_issue(f"other {i}", f"2026-11-19T10:{i:02d}:00Z")
    gh.add_issue("quoted <!-- report_id: rpt_00000000000000b2 --> mid-body", "2026-11-19T11:00:00Z")
    for i in range(15):  # older issues, commented on lately: listed after the marked one
        gh.add_issue(f"old {i}", f"2026-11-10T10:{i:02d}:00Z", updated_at="2026-11-19T11:30:00Z")
    a = _put(reports, 1, _record("rpt_00000000000000a1"))
    _put(reports, 2, _record("rpt_00000000000000b2"))
    assert _main(reports, gh) == 0
    assert _processed(reports, a)["issue_number"] == marked["number"]
    assert len(gh.posted()) == 1  # b2's marker was not at the end of a body
    scans = [c for c in gh.calls if c[:2] == ("GET", f"/repos/{REPO}/issues")]
    assert len(scans) == 3  # the third page reached the old issues; no fourth asked for
    assert all(c[3]["Authorization"] == "Bearer ghp_testtoken" for c in scans)


def test_scan_starts_an_hour_before_the_oldest_report():
    since = fi._scan_from([_name(5, "rpt_00000000000000a1"), "junk.json", _name(3, "rpt_00000000000000b2")])
    assert since.strftime("%Y-%m-%dT%H:%M:%SZ") == "2026-11-19T07:03:00Z"


@pytest.mark.parametrize("status,headers", [
    (403, {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1795000000"}),
    (403, {"Retry-After": "60"}),
    (429, {}),
])
def test_rate_limit_stops_and_leaves_the_rest(reports, capsys, status, headers):
    gh = FakeGitHub()
    names = [_put(reports, i, _record(f"rpt_0000000000000{i}a0")) for i in range(1, 4)]
    gh.fail[("POST", f"/repos/{REPO}/issues")] = [
        (201, {}, {"number": 1, "html_url": "https://github.com/x/1"}),
        (status, headers, {"message": "You have exceeded a secondary rate limit"}),
    ]
    assert _main(reports, gh) == 0
    assert _inbox(reports) == names[1:]
    assert len(gh.posted()) == 2  # the refused one and nothing after it
    out = capsys.readouterr().out
    assert "stopped by GitHub's rate limit" in out
    assert out.strip().splitlines()[-1] == "posted 1, skipped 0, failed 0, left 2"


def test_a_refused_issue_fails_alone(reports):
    gh = FakeGitHub()
    names = [_put(reports, i, _record(f"rpt_0000000000000{i}a0")) for i in range(1, 3)]
    gh.fail[("POST", f"/repos/{REPO}/issues")] = [(422, {}, {"message": "Validation Failed"})]
    assert _main(reports, gh) == 1
    assert _inbox(reports) == names[:1] and len(gh.issues) == 1


@pytest.mark.parametrize("status", [401, 403, 404, 500, 502])
def test_other_errors_stop_the_run_as_failed(reports, capsys, status):
    gh = FakeGitHub()
    names = [_put(reports, i, _record(f"rpt_0000000000000{i}a0")) for i in range(1, 3)]
    gh.fail[("POST", f"/repos/{REPO}/issues")] = [(status, {}, {"message": "nope"})]
    assert _main(reports, gh) == 1
    assert _inbox(reports) == names
    out = capsys.readouterr().out
    assert f"HTTP {status} nope" in out and out.strip().endswith("left 2")


def test_a_network_failure_stops_the_run_as_failed(reports):
    def down(method, url, headers, data):
        raise OSError("connection refused")

    name = _put(reports, 1, _record())
    assert _main(reports, down) == 1
    assert _inbox(reports) == [name]


def test_writes_are_paced(reports):
    gh = FakeGitHub(labels=list(LABELS))
    for i in range(1, 4):
        _put(reports, i, _record(f"rpt_0000000000000{i}a0"))
    sleeps: list[float] = []
    assert _main(reports, gh, sleeps=sleeps) == 0
    assert sleeps == [fi.PAUSE_SECONDS] * 2  # before the 2nd and the 3rd issue


def test_an_empty_inbox_makes_no_call(reports, capsys):
    gh = FakeGitHub()
    assert _main(reports, gh) == 0
    assert gh.calls == []
    assert capsys.readouterr().out.strip() == "posted 0, skipped 0, failed 0, left 0"


# --- token and dry run ---------------------------------------------------------

def test_the_token_comes_from_stdin_and_is_never_printed(reports, capsys):
    gh = FakeGitHub()
    _put(reports, 1, _record())
    gh.fail[("POST", f"/repos/{REPO}/issues")] = [(401, {}, {"message": "Bad credentials"})]
    assert _main(reports, gh, token="ghp_secret123\n") == 1
    assert {c[3]["Authorization"] for c in gh.calls} == {"Bearer ghp_secret123"}
    captured = capsys.readouterr()
    assert "ghp_secret123" not in captured.out + captured.err


@pytest.mark.parametrize("token", ["", "  \n", "two words", "ghp_a\nghp_b"])
def test_a_missing_or_odd_token_exits_before_any_call(reports, capsys, token):
    gh = FakeGitHub()
    _put(reports, 1, _record())
    assert _main(reports, gh, token=token) == 2
    assert gh.calls == []
    out = capsys.readouterr().out
    assert "error:" in out and (not token.strip() or token.split()[0] not in out)


def test_dry_run_needs_no_token_and_no_network(reports, tmp_path, capsys):
    class NoStdin(io.StringIO):
        def read(self, *a):
            raise AssertionError("dry run read stdin")

    def no_network(*a):
        raise AssertionError("dry run called GitHub")

    name = _put(reports, 1, _record(problem_summary="see <module>"))
    out_dir = tmp_path / "preview"
    assert fi.main(["--dry-run", "--reports-dir", str(reports), "--out", str(out_dir)],
                   stdin=NoStdin(), transport=no_network) == 0
    preview = (out_dir / name.replace(".json", ".md")).read_text(encoding="utf-8")
    assert preview.startswith("# see &lt;module&gt;\n\nLabels: `area:docs` `target:how-to`\n\n### Problem")
    assert _inbox(reports) == [name]  # nothing moved
    assert fi.main(["--dry-run", "--reports-dir", str(reports)], stdin=NoStdin(), transport=no_network) == 0
    assert "=== " + name in capsys.readouterr().out
