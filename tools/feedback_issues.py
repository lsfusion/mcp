"""Turn stored feedback reports into issues in lsfusion/platform-feedback.

`lsfusion_report_feedback` (tools/feedback.py) stores each report as its own
file, `<REPORTS_DIR>/inbox/<UTC YYYYMMDDTHHMMSS>-<report_id>.json`. This worker
takes those files oldest first and, for each one, opens one issue, writes the
record with `issue_number` and `issue_url` added to `<REPORTS_DIR>/processed/`,
and removes it from the inbox.

It runs inside the mcp container, started over SSH. The token comes on stdin,
so it is never on a command line:

    docker exec -i stack-mcp-1 python -m tools.feedback_issues < /etc/lsfusion-feedback/github-token

`--dry-run` renders the issues instead, with no token and no network call;
`--out DIR` writes one Markdown file per report there. Standard library only.

No duplicates: before posting, the worker lists the repo's issues created since
the oldest pending report and reads the `<!-- report_id: ... -->` marker each
body ends with. That list is read from the issues endpoint, not the search
index, so it is never stale. A report whose issue already exists, because an
earlier run died between posting and moving the file, is only moved.

Agent-written text never pings or links anything: mentions, issue references,
URLs and link brackets get a zero-width space that breaks them, and HTML is
escaped outside code.
"""
from __future__ import annotations

import argparse
import html
import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

from settings import REPORTS_DIR

# Fixed, not an option: the token (a classic PAT with `repo` scope) reaches
# every repo its user can, and this worker has business only in this one.
REPO = "lsfusion/platform-feedback"
API = "https://api.github.com"
# Between two writes. GitHub's secondary limits punish bursts of content
# creation (80 a minute, 500 an hour); a run of a day's ~200 reports at this
# pace stays well inside both.
PAUSE_SECONDS = 2.0
TITLE_MAX = 100
# GitHub refuses a body over 65536 characters; this leaves a margin.
BODY_MAX = 65000
EVAL_ERRORS_SHOWN = 10  # the rest are in the JSON
# The scan for already-posted reports starts this long before the oldest
# pending one, for clock skew between this server and GitHub.
SCAN_MARGIN = timedelta(hours=1)

# primary_target -> area: who acts on the report.
AREAS = {
    "language": "docs", "how-to": "docs", "paradigm": "docs", "rules": "docs", "brief": "docs",
    "code-bug": "platform", "eval-error-message": "platform",
    "rag-retrieval": "search",
}

# Every issue the worker opens carries it: an agent's report as received,
# primary data that stays unedited. The work it leads to gets issues of its own.
RAW = "raw"

# The worker's labels, created when missing. Agents read the descriptions with
# `gh label list`, so each one says what the label means.
LABELS = {
    RAW: ("ededed", "An agent's feedback report as received: primary data, kept unedited"),
    "area:docs": ("1d76db", "Fix belongs in the docs: language reference, how-to, paradigm, rules or brief"),
    "area:platform": ("b60205", "Fix belongs in the platform: a code bug or an unclear eval error message"),
    "area:search": ("5319e7", "Fix belongs in doc search: retrieve_docs found the wrong articles or none"),
    "target:language": ("c5def5", "Report's primary_target: the language reference"),
    "target:how-to": ("c5def5", "Report's primary_target: a how-to article"),
    "target:paradigm": ("c5def5", "Report's primary_target: a paradigm article"),
    "target:rules": ("c5def5", "Report's primary_target: the Rules guidance article (get_guidance)"),
    "target:brief": ("c5def5", "Report's primary_target: the Brief guidance article (get_guidance)"),
    "target:code-bug": ("f9d0c4", "Report's primary_target: a bug or missing feature in the platform code"),
    "target:eval-error-message": ("f9d0c4", "Report's primary_target: an eval error message (wording, location, hint)"),
    "target:rag-retrieval": ("d4c5f9", "Report's primary_target: doc search ranking or coverage (retrieve_docs)"),
}

_MARKER = "<!-- report_id: {} -->"
# Only a marker that ENDS the body counts, so a marker quoted inside a report's
# own text can never stand in for its issue.
_MARKER_RE = re.compile(r"<!-- report_id: (rpt_[A-Za-z0-9_]+) -->\s*\Z")
_REPORT_ID = re.compile(r"rpt_[A-Za-z0-9_]+")
_NAME = re.compile(r"(\d{8}T\d{6})-rpt_[A-Za-z0-9_]+\.json")
_TITLE_TAGS = re.compile(r"(?:\[[^\[\]]{1,40}\]\s*)+")


# --- making agent text safe to post ------------------------------------------

ZWSP = "​"
# What GitHub turns into a ping or a link: @mentions (and emails), #123 and
# GH-123 references, URLs (scheme:// and www.), and the brackets of links,
# images and reference definitions. A zero-width space after the trigger breaks
# each one. It goes into code spans too: they need no protection, but finding
# them exactly as GitHub does is not worth betting a ping on, and the space
# shows nowhere.
_UNLINK = re.compile(r"@(?=[A-Za-z0-9])|#(?=\d)|\bGH-(?=\d)|:(?=//)|\bwww(?=\.)|\](?=[(\[:])", re.I)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# An inline code span within one line, as CommonMark pairs them: a backtick run
# not escaped by a backslash, closed by the next run of exactly its length.
_SPAN = re.compile(r"(?<![`\\])(`+)(?!`)(.+?)(?<!`)\1(?!`)")
_FENCE_OPEN = re.compile(r" {0,3}(`{3,}|~{3,})(.*)")


def _unlink(s: str) -> str:
    return _UNLINK.sub(lambda m: m.group(0) + ZWSP, s)


def _clean(value) -> str:
    """A record field as text: newlines normalized, control characters dropped."""
    if value is None:
        return ""
    s = value if isinstance(value, str) else str(value)
    return _CONTROL.sub("", s.replace("\r\n", "\n").replace("\r", "\n"))


def _escape_line(line: str) -> str:
    """HTML-escape a line of prose, leaving its inline code spans as written.
    A line with a pipe may be a table row, where GitHub splits the cells before
    it looks for code spans, so all of it is escaped."""
    if "|" in line:
        return html.escape(line, quote=False)
    out, pos = [], 0
    for m in _SPAN.finditer(line):
        out.append(html.escape(line[pos:m.start()], quote=False))
        out.append(m.group(0))
        pos = m.end()
    out.append(html.escape(line[pos:], quote=False))
    return "".join(out)


def _md(text: str) -> str:
    """Agent-written Markdown made safe to post. It renders as written, but
    pings nobody, links nothing and carries no HTML, and a code fence it leaves
    open is closed, so it cannot swallow the rest of the issue.

    Callers pass the whole first line, label included (`**Why:** ...`), so a
    fence is recognised only where GitHub would recognise it too."""
    out, fence = [], None
    for line in _unlink(text).split("\n"):
        if fence:
            out.append(line)
            if re.fullmatch(r" {0,3}%s{%d,}[ \t]*" % (re.escape(fence[0]), len(fence)), line):
                fence = None
            continue
        m = _FENCE_OPEN.fullmatch(line)
        if m and not (m.group(1)[0] == "`" and "`" in m.group(2)):
            fence = m.group(1)
            out.append(line)
        else:
            out.append(_escape_line(line))
    if fence:
        out.append(fence)
    return "\n".join(out)


def _fence(text: str, info: str = "") -> str:
    """`text` verbatim in a code block whose fence no backtick run inside can close."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    bar = "`" * max(3, longest + 1)
    return f"{bar}{info}\n{text}\n{bar}"


def _cut(s: str, limit: int | None) -> str:
    """`s` up to `limit` characters, cut at a word, with an ellipsis if cut."""
    if limit is None or len(s) <= limit:
        return s
    head = s[:limit + 1]
    space = head.rfind(" ")
    head = head[:space] if space >= limit // 2 else s[:limit]
    return head.rstrip(" \n,.;:-–—") + "…"


# --- rendering ---------------------------------------------------------------

def render_title(rec: dict) -> str:
    """The start of problem_summary, cut at a word. Leading tags such as
    `[CONVEYOR][S3]` are left to the body, where the summary is whole: in a
    title they would take half the room. Titles show no HTML, so only pings
    and references need breaking."""
    summary = " ".join(_clean(rec.get("problem_summary")).split())
    tags = _TITLE_TAGS.match(summary)
    if tags and tags.end() < len(summary):
        summary = summary[tags.end():]
    return _unlink(_cut(summary, TITLE_MAX)) or f"Feedback report {rec['report_id']}"


def _who(rec: dict) -> str:
    def flat(key):
        return " ".join(_clean(rec.get(key)).split())

    reported = "Reported"
    if flat("agent"):
        reported += f" by {flat('agent')}"
    if flat("client"):
        reported += f" via {' '.join(filter(None, (flat('client'), flat('client_version'))))}"
    # When the server received it: the issue's own date is when the worker ran,
    # an hour later at most for a live report, months for one carried over.
    when = re.match(r"(\d{4}-\d\d-\d\d)T(\d\d:\d\d)", flat("ts"))
    if when:
        reported += f" on {when[1]} {when[2]} UTC"
    parts = [reported] if reported != "Reported" else []
    if flat("ide_build"):  # only the IntelliJ plugin's reports carry it
        parts.append(f"IDE {flat('ide_build')}")
    if flat("lsfusion_version"):
        parts.append(f"lsFusion {flat('lsfusion_version')}")
    server = re.sub(r"[^\w.-]", "", flat("server_version"))[:40]
    if server:
        if re.fullmatch(r"[0-9a-f]{12,40}", server):
            server = server[:7]
        parts.append(f"MCP server `{server}`")
    return " · ".join(parts)


def _sections(rec: dict, cap: int | None, errors_shown: int) -> str:
    def text(value):
        return _cut(_clean(value).strip(), cap)

    rec_ = rec.get("recommendation") or {}
    exp = rec.get("expectation") if isinstance(rec.get("expectation"), dict) else {}
    blocks = ["### Problem", _md(text(rec.get("problem_summary")))]

    expected, actual = text(exp.get("expected")), text(exp.get("actual"))
    if expected or actual:
        blocks.append("### Expected vs actual")
        if expected:
            blocks.append(_md("**Expected:** " + expected))
        if actual:
            blocks.append(_md("**Actual:** " + actual))

    blocks.append("### Suggested change")
    if text(rec_.get("suggested_change")):
        blocks.append(_md(text(rec_.get("suggested_change"))))
    if text(rec_.get("rationale")):
        blocks.append(_md("**Why:** " + text(rec_.get("rationale"))))
    facts = "**Confidence:** " + (text(rec_.get("confidence")) or "unknown")
    also = [t for t in (text(x) for x in rec_.get("secondary_targets") or []) if t]
    if also:
        facts += " · **Also:** " + ", ".join(also)
    blocks.append(_md(facts))

    errors = [e for e in rec.get("eval_errors") or [] if isinstance(e, dict)]
    if errors:
        blocks.append("### Eval errors")
        for e in errors[:errors_shown]:
            blocks.append(_md(f"**{text(e.get('phase')) or 'unknown'}** · " + text(e.get("message"))))
            code = _cut(_clean(e.get("code_excerpt")).strip("\n").rstrip(), cap)
            if code:
                blocks.append(_fence(code))
        if len(errors) > errors_shown:
            blocks.append(f"…and {len(errors) - errors_shown} more in the JSON below.")

    blocks.append(_md(f"**Outcome:** {text(rec.get('final_outcome')) or 'unknown'}"
                      f" · **Signal:** {text(rec.get('signal_type')) or 'unknown'}"))
    who = _who(rec)
    if who:
        blocks.append(_md(who))
    return "\n\n".join(blocks)


def _assemble(head: str, json_text: str, note: str | None, marker: str) -> str:
    summary = "Full report (JSON)" if note is None else "Report (JSON, cut)"
    details = f"<details>\n<summary>{summary}</summary>\n\n{_fence(json_text, 'json')}\n"
    if note:
        details += f"\n{note}\n"
    return f"{head}\n\n{details}\n</details>\n\n{marker}\n"


def render_body(rec: dict, name: str) -> str:
    """The issue body: short sections, the full record in a collapsed block,
    and the report_id marker as its last line. Over GitHub's limit the JSON is
    cut first; only if that is not enough are the sections shortened too."""
    marker = _MARKER.format(rec["report_id"])
    full = json.dumps(rec, ensure_ascii=False, indent=2)
    for cap, shown in ((None, EVAL_ERRORS_SHOWN), (500, 3)):
        head = _sections(rec, cap, shown)
        body = _assemble(head, full, None, marker)
        if len(body) <= BODY_MAX:
            return body
        keep = len(full) - (len(body) - BODY_MAX) - 300  # room for the note
        if keep >= 1000:
            cut = full[:keep].rsplit("\n", 1)[0] + "\n…"
            note = (f"Cut to fit GitHub's size limit: {len(cut)} of {len(full)} characters. "
                    f"The whole report is kept on the server as `processed/{name}`.")
            body = _assemble(head, cut, note, marker)
            if len(body) <= BODY_MAX:
                return body
    return f"{head}\n\n{marker}\n"  # not reached: the shortened sections always leave room


def labels_for(rec: dict) -> list[str]:
    target = rec["recommendation"]["primary_target"]
    return [RAW, f"area:{AREAS[target]}", f"target:{target}"]


def render_issue(rec: dict, name: str) -> dict:
    return {"title": render_title(rec), "body": render_body(rec, name), "labels": labels_for(rec)}


# --- the inbox ---------------------------------------------------------------

class ReportError(Exception):
    """A file in the inbox that cannot become an issue. It stays where it is."""


def pending(inbox: str) -> list[str]:
    """The report files waiting in `inbox`, oldest first. Dot files are the
    server's half-written temp files and are never touched."""
    try:
        names = os.listdir(inbox)
    except FileNotFoundError:
        return []
    return sorted(n for n in names
                  if n.endswith(".json") and not n.startswith(".")
                  and os.path.isfile(os.path.join(inbox, n)))


def load_report(inbox: str, name: str) -> dict:
    if not _NAME.fullmatch(name):
        raise ReportError("unexpected file name")
    try:
        with open(os.path.join(inbox, name), encoding="utf-8") as f:
            rec = json.load(f)
    except (OSError, ValueError) as e:
        raise ReportError(f"unreadable: {type(e).__name__}: {e}") from None
    if not isinstance(rec, dict):
        raise ReportError("not a JSON object")
    if not isinstance(rec.get("report_id"), str) or not _REPORT_ID.fullmatch(rec["report_id"]):
        raise ReportError("no valid report_id")
    target = rec["recommendation"].get("primary_target") if isinstance(rec.get("recommendation"), dict) else None
    if target not in AREAS:
        raise ReportError(f"unknown primary_target {target!r}")
    return rec


def _write_json(path: str, record: dict) -> None:
    """Write `record` all at once or not at all, as the server writes the inbox:
    to a dot-named temp file in the same folder, then renamed into place."""
    folder, name = os.path.split(path)
    os.makedirs(folder, exist_ok=True)
    tmp = os.path.join(folder, "." + name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _processed_issue(reports_dir: str, name: str):
    """(number, url) if `processed/` already holds this file with its issue."""
    try:
        with open(os.path.join(reports_dir, "processed", name), encoding="utf-8") as f:
            prev = json.load(f)
    except (OSError, ValueError):
        return None
    number = prev.get("issue_number") if isinstance(prev, dict) else None
    return (number, prev.get("issue_url")) if isinstance(number, int) else None


def _file_away(reports_dir: str, name: str, rec: dict, number: int, url: str) -> None:
    _write_json(os.path.join(reports_dir, "processed", name),
                {**rec, "issue_number": number, "issue_url": url})
    os.remove(os.path.join(reports_dir, "inbox", name))


def _scan_from(names: list[str]) -> datetime | None:
    times = []
    for name in names:
        m = _NAME.fullmatch(name)
        if m:
            try:
                times.append(datetime.strptime(m.group(1), "%Y%m%dT%H%M%S"))
            except ValueError:
                pass
    return (min(times) - SCAN_MARGIN).replace(tzinfo=timezone.utc) if times else None


# --- GitHub ------------------------------------------------------------------

class Stop(Exception):
    """The run cannot go on: the rest stays in the inbox, and the exit code is 1."""


class GitHubError(Stop):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class RateLimited(Exception):
    """GitHub asked us to slow down: stop, and leave the rest for the next run."""


def urllib_transport(method: str, url: str, headers: dict, data: bytes | None):
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, dict(resp.headers.items()), resp.read()
    except urllib.error.HTTPError as err:
        return err.code, dict(err.headers.items()) if err.headers else {}, err.read()


def _next_link(headers: dict) -> str | None:
    for part in headers.get("link", "").split(","):
        m = re.search(r'<([^>]+)>\s*;\s*rel="next"', part)
        if m:
            return m.group(1)
    return None


class GitHub:
    """A thin REST client. Writes are paced; the token goes only to the API host."""

    def __init__(self, token: str, transport=urllib_transport, *,
                 pause: float = PAUSE_SECONDS, sleep=time.sleep, clock=time.monotonic):
        self._token = token
        self._transport = transport
        self._pause = pause
        self._sleep = sleep
        self._clock = clock
        self._last_write = None

    def call(self, method: str, path: str, body=None):
        url = path if path.startswith("https://") else API + path
        if not url.startswith(API + "/"):
            raise Stop(f"refusing to send the token outside {API}")
        where = f"{method} {url[len(API):]}"
        if method != "GET" and self._last_write is not None:
            wait = self._last_write + self._pause - self._clock()
            if wait > 0:
                self._sleep(wait)
        headers = {
            "Authorization": "Bearer " + self._token,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "lsfusion-feedback-issues",
        }
        data = None
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        try:
            status, resp_headers, raw = self._transport(method, url, headers, data)
        except (OSError, http.client.HTTPException) as e:
            raise GitHubError(f"{where}: {type(e).__name__}: {e}") from None
        finally:
            if method != "GET":
                self._last_write = self._clock()
        resp_headers = {k.lower(): v for k, v in resp_headers.items()}
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else None
        except ValueError:
            payload = None
        message = payload.get("message", "") if isinstance(payload, dict) else ""
        if status == 429 or (status == 403 and (
                "retry-after" in resp_headers
                or resp_headers.get("x-ratelimit-remaining") == "0"
                or "rate limit" in message.lower())):
            raise RateLimited(_retry_hint(resp_headers))
        if status >= 400:
            detail = f"{where}: HTTP {status} {message}".rstrip()
            if isinstance(payload, dict) and payload.get("errors"):
                detail += " " + json.dumps(payload["errors"], ensure_ascii=False)[:300]
            raise GitHubError(detail, status)
        return payload, resp_headers


def _retry_hint(headers: dict) -> str:
    if headers.get("retry-after"):
        return f"retry after {headers['retry-after']} s"
    if headers.get("x-ratelimit-reset", "").isdigit():
        reset = datetime.fromtimestamp(int(headers["x-ratelimit-reset"]), timezone.utc)
        return f"resets at {reset:%H:%M:%S} UTC"
    return "no retry time given"


def ensure_labels(gh: GitHub, repo: str) -> list[str]:
    """Create those of LABELS the repo lacks. Existing labels, GitHub's
    defaults among them, are never changed."""
    have, path = set(), f"/repos/{repo}/labels?per_page=100"
    while path:
        items, headers = gh.call("GET", path)
        have |= {item["name"].lower() for item in items}
        path = _next_link(headers)
    created = []
    for name, (color, description) in LABELS.items():
        if name.lower() in have:
            continue
        try:
            gh.call("POST", f"/repos/{repo}/labels",
                    {"name": name, "color": color, "description": description})
        except GitHubError as e:
            if e.status != 422:  # 422: created meanwhile
                raise
            continue
        created.append(name)
    return created


def posted_reports(gh: GitHub, repo: str, since: datetime | None) -> dict:
    """{report_id: (number, url)} for the issues created since `since` whose
    body ends with a report marker. Read from the issues list, newest first,
    which (unlike search) shows an issue as soon as it is created."""
    if since is None:
        return {}
    since_iso = since.strftime("%Y-%m-%dT%H:%M:%SZ")
    found = {}
    path = f"/repos/{repo}/issues?" + urllib.parse.urlencode({
        "state": "all", "sort": "created", "direction": "desc", "since": since_iso, "per_page": 100})
    while path:
        items, headers = gh.call("GET", path)
        for item in items:
            if item.get("created_at", "") < since_iso:
                return found
            m = _MARKER_RE.search(item.get("body") or "")
            if m and "pull_request" not in item:
                found.setdefault(m.group(1), (item["number"], item["html_url"]))
        path = _next_link(headers)
    return found


# --- the runs ----------------------------------------------------------------

def run(reports_dir: str, gh: GitHub, repo: str) -> int:
    names = pending(os.path.join(reports_dir, "inbox"))
    posted = skipped = failed = 0
    left = len(names)
    stopped = None
    code = 0
    if names:
        try:
            for label in ensure_labels(gh, repo):
                print(f"created label {label}", flush=True)
            on_github = posted_reports(gh, repo, _scan_from(names))
            for name in names:
                try:
                    rec = load_report(os.path.join(reports_dir, "inbox"), name)
                except ReportError as e:
                    print(f"failed {name}: {e}", flush=True)
                    failed, left = failed + 1, left - 1
                    continue
                done = _processed_issue(reports_dir, name)
                if done:
                    os.remove(os.path.join(reports_dir, "inbox", name))
                elif rec["report_id"] in on_github:
                    done = on_github[rec["report_id"]]
                    _file_away(reports_dir, name, rec, *done)
                if done:
                    print(f"skipped {name}: already #{done[0]}", flush=True)
                    skipped, left = skipped + 1, left - 1
                    continue
                try:
                    issue, _ = gh.call("POST", f"/repos/{repo}/issues", render_issue(rec, name))
                except GitHubError as e:
                    if e.status != 422:  # 422: GitHub refused this one issue; the rest may go
                        raise
                    print(f"failed {name}: {e}", flush=True)
                    failed, left = failed + 1, left - 1
                    continue
                try:
                    _file_away(reports_dir, name, rec, issue["number"], issue["html_url"])
                except OSError as e:
                    raise Stop(f"posted #{issue['number']} for {name}, but could not move the file: {e}") from None
                print(f"posted #{issue['number']} {name}", flush=True)
                posted, left = posted + 1, left - 1
        except RateLimited as e:
            stopped = f"stopped by GitHub's rate limit ({e})"
        except (Stop, OSError) as e:
            stopped, code = f"error: {e}", 1
    if stopped:
        print(stopped, flush=True)
    print(f"posted {posted}, skipped {skipped}, failed {failed}, left {left}", flush=True)
    return 1 if code or failed else 0


def _preview(issue: dict) -> str:
    labels = " ".join(f"`{label}`" for label in issue["labels"])
    return f"# {html.escape(issue['title'], quote=False)}\n\nLabels: {labels}\n\n{issue['body']}"


def dry_run(reports_dir: str, out_dir: str | None) -> int:
    inbox = os.path.join(reports_dir, "inbox")
    names = pending(inbox)
    failed = 0
    for name in names:
        try:
            issue = render_issue(load_report(inbox, name), name)
        except ReportError as e:
            print(f"failed {name}: {e}")
            failed += 1
            continue
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, name[:-len(".json")] + ".md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(_preview(issue))
            print(f"rendered {path}")
        else:
            print(f"=== {name}\n{_preview(issue)}")
    print(f"dry run: rendered {len(names) - failed}, failed {failed}")
    return 1 if failed else 0


def main(argv=None, *, stdin=None, transport=urllib_transport,
         sleep=time.sleep, clock=time.monotonic) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.feedback_issues",
        description="Post each report in <REPORTS_DIR>/inbox as an issue. "
                    "The GitHub token is read from stdin.")
    parser.add_argument("--dry-run", action="store_true",
                        help="render the issues; no token, no network")
    parser.add_argument("--out", metavar="DIR",
                        help="with --dry-run: write one Markdown file per report into DIR")
    parser.add_argument("--reports-dir", default=REPORTS_DIR,
                        help="default: REPORTS_DIR (%(default)r)")
    args = parser.parse_args(argv)
    if args.out and not args.dry_run:
        parser.error("--out goes with --dry-run")
    if not args.reports_dir:
        print("error: no reports dir: set REPORTS_DIR or LOG_DIR, or pass --reports-dir")
        return 2
    if args.dry_run:
        return dry_run(args.reports_dir, args.out)

    stdin = sys.stdin if stdin is None else stdin
    token = "" if stdin.isatty() else stdin.read().strip()
    if not token:
        print("error: no GitHub token on stdin")
        return 2
    if not re.fullmatch(r"[\x21-\x7e]+", token):
        # Never echo it: a bad header value would put it in a traceback.
        print("error: the token on stdin is not a single word of printable ASCII")
        return 2
    gh = GitHub(token, transport, sleep=sleep, clock=clock)
    return run(args.reports_dir, gh, REPO)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
