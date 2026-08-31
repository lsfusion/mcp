"""get_guidance source — fetch a published Brief/Rules article WHOLE.

A thin proxy over the docs site's per-page Markdown twins, emitted by
`@signalwire/docusaurus-plugin-llms-txt` at `/<slug>.md` (e.g.
https://docs.lsfusion.org/Brief.md, https://docs.lsfusion.org/Rules_logic.md).
The published site is the single source of truth: no cache, no packaged
fallback. If a page is unreachable the call fails (the timeout prevents a hang)
rather than returning stale guidance.

The `brief` and `rules` branches are NOT a search corpus. Relevance there is
not probabilistic: a rules article is only useful whole, and a top-N chunk
retrieval cannot say which chunk it withheld — an assistant that received 3 of
an article's 4 chunks has no way to know the 4th existed. So these two branches
are addressed by NAME and delivered entire; `retrieve_docs` keeps the branches
where ranked excerpts are the right answer.

Kept free of FastMCP/SDK imports so it is unit-testable on its own; `server.py`
just wires `lsfusion_get_guidance` to `fetch_guidance()`.
"""
from __future__ import annotations

import hashlib
import os
import re
import urllib.error
import urllib.request

GUIDANCE_BASE_URL = os.getenv("GUIDANCE_BASE_URL", "https://docs.lsfusion.org/")
BRIEF_URL = os.getenv("GUIDANCE_BRIEF_URL", f"{GUIDANCE_BASE_URL}Brief.md")
RULES_URL = os.getenv("GUIDANCE_RULES_URL", f"{GUIDANCE_BASE_URL}Rules.md")
FETCH_TIMEOUT = float(os.getenv("GUIDANCE_FETCH_TIMEOUT", "10"))

# The two branches served by name, and the slug prefix each one's articles use.
BRANCH_PREFIX = {"rules": "Rules", "brief": "Brief"}

# An area name, as written in the map inside the branch's top article. Anchored
# and deliberately narrow: the prefix is supplied by us, so no name a caller
# sends can escape its branch, name a page in another branch, or walk the path
# (no `/`, no `.`, hence no dot-segments). Names that reach the site are always
# `Rules_<area>.md` or `Brief_<area>.md`.
AREA_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")

# Names that mean "the branch's own top article" rather than one of its areas.
TOP_ALIASES = frozenset({"top", "rules", "brief", "index", "map"})

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
    "`lsfusion_get_guidance` with no arguments and apply every rule it returns "
    "according to that rule's stated strength (MUST / MUST NOT are binding; "
    "SHOULD / SHOULD NOT are recommendations). Once per session is enough. If "
    "your client saves the result to a file and shows only a preview, read "
    "that file in full before continuing.\n\n"
    "What that returns is the TOP article of each branch: the base material "
    "plus a complete map of the branch's other articles. An area's rules are a "
    "separate article, read WHOLE with `lsfusion_get_guidance(rules='<area>')` "
    "— never searched, never excerpted. The map states, per area, the point at "
    "which reading it stops being optional; an area you did not read is not an "
    "area without rules.\n\n"
    "Other tools: `lsfusion_retrieve_docs` searches the reference "
    "documentation (`language`, `paradigm`, `how-to`); "
    "`lsfusion_report_feedback` submits a consented, depersonalized quality "
    "signal (the guidance says when)."
)

# First lines of every `lsfusion_get_guidance` result. A client that persists an
# oversized tool result to a file still shows the assistant a preview of the
# head, so this notice is the one part guaranteed to be read — spend it telling
# the assistant how to recover the part it cannot see, and where the rules that
# are NOT in this response live.
GUIDANCE_NOTICE = (
    "> IMPORTANT — this message contains the TOP article of each guidance "
    "branch, each one whole between its `=== BEGIN ... ===` and "
    "`=== END ... ===` lines: the base material plus the complete map of every "
    "other article in that branch. It is not every rule that applies. An area's rules are a "
    "separate article, read whole with `lsfusion_get_guidance(rules='<area>')` "
    "using a name from the map below; the map states when each one becomes "
    "mandatory. A summary line in the map is an index entry, not the rule — do "
    "not infer what an article says from it, and do not treat an area you did "
    "not read as an area without rules. If your client truncated this message, "
    "or saved it to a file and showed you only a preview, read that file IN "
    "FULL before writing or reviewing any lsFusion code. Apply each rule at "
    "its stated strength: MUST / MUST NOT are binding, SHOULD / SHOULD NOT are "
    "recommendations."
)


def _fetch(url: str, timeout: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (https only, fixed hosts)
        return resp.read().decode("utf-8")


def fetch_guidance(urls: list[str] | None = None, timeout: float | None = None) -> str:
    """Fetch each URL and return the concatenated text. Exceptions propagate."""
    urls = [BRIEF_URL, RULES_URL] if urls is None else urls
    timeout = FETCH_TIMEOUT if timeout is None else timeout
    return "\n\n".join(_fetch(u, timeout) for u in urls)


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
    """The top article of each branch, each one fenced, headed by the notice.

    Fenced for the same reason a named article is, and this is the call that
    needs it most: it is the one an assistant is told to make on every task, it
    is the largest single result this server returns, and it carries the maps —
    so a silent truncation here costs the assistant not one article but its
    knowledge that the others exist.

    The version stamps the concatenated BODIES alone, so it equals an
    independent hash of the published Brief+Rules and stays stable as the
    fences and the notice wording evolve.
    """
    urls = [BRIEF_URL, RULES_URL] if urls is None else urls
    timeout = FETCH_TIMEOUT if timeout is None else timeout
    bodies = [_fetch(u, timeout) for u in urls]
    blocks = [fenced(_branch_of(u), "top", b) for u, b in zip(urls, bodies)]
    body = "\n\n".join(bodies)
    return f"{_version_marker(body)}\n{GUIDANCE_NOTICE}\n\n" + "\n\n".join(blocks)


def _branch_of(url: str) -> str:
    """Which branch a top-article URL belongs to, for its fence label."""
    tail = url.rsplit("/", 1)[-1].lower()
    for branch, prefix in BRANCH_PREFIX.items():
        if tail.startswith(prefix.lower()):
            return branch
    return "guidance"


# ---------------------------------------------------------------------------
# Reading ONE article by name
# ---------------------------------------------------------------------------
#
# The result is fenced, and the fence is load-bearing. A header field asserting
# `complete: yes` survives truncation and then lies — the assistant reads a
# claim of completeness sitting on top of a body that lost its tail, which is
# exactly the failure this delivery path exists to remove. A TERMINATOR cannot
# lie: truncation takes it with the tail. So completeness is proved by the END
# fence being present, and the char count is repeated in both fences so a loss
# in the middle is catchable too.
ARTICLE_BEGIN = "=== BEGIN lsfusion {branch}/{name} | rev {rev} | chars {n} ==="
ARTICLE_END = "=== END lsfusion {branch}/{name} | chars {n} ==="

ARTICLE_NOTICE = (
    "> This is ONE COMPLETE lsFusion guidance article, delivered whole — not a "
    "search result and not an excerpt. The `=== END lsfusion {branch}/{name} "
    "... ===` line at the very bottom is what shows it was not cut short: if "
    "it is missing, the copy was truncated in transit. Its `chars` is the "
    "article's length as sent, so if what you are holding is visibly shorter "
    "than that, you are holding a preview — and if your client saved this "
    "result to a file, read that file IN FULL before applying anything from "
    "it. Apply each rule at its "
    "stated strength: MUST / MUST NOT are binding, SHOULD / SHOULD NOT are "
    "recommendations. Nothing else was read: the other articles of this branch "
    "are listed in the map inside its top article, which "
    "`lsfusion_get_guidance()` with no arguments returns."
)

# Answer to a name that does not resolve. Deliberately a tool SUCCESS carrying
# this text, never an MCP error: in these harnesses a failed tool call is about
# as likely to produce "the rules lookup did not work, I'll carry on" as a
# retry, whereas a success that hands over the recovery step makes the correct
# second call the path of least resistance. The one sentence that must survive
# any shortening of this text is the third: nothing was read, so nothing was
# ruled out.
NOT_FOUND_NOTICE = (
    "> NO SUCH GUIDANCE ARTICLE — you asked for {branch}='{name}'.\n"
    "> The `{branch}` branch has no article by that name{detail}.\n"
    "> NOTHING WAS READ. This is not a finding that no {branch} article covers "
    "your area, and it is not a finding that no rules apply to it: no article "
    "was fetched, so no conclusion about your area follows from this result.\n"
    "> Call `lsfusion_get_guidance()` with NO arguments — the top article of "
    "each branch carries the complete map of that branch's articles — then "
    "call `lsfusion_get_guidance({branch}='<name>')` with a name from the map. "
    "Names are the short area names in the map's first column: not slugs "
    "(`Rules_logic`), not titles (\"Rules: domain logic\"), not file names."
)


def article_slug(branch: str, name: str) -> str:
    """Published slug for one guidance article, or raise ValueError.

    `branch` is ours (a bug if wrong); `name` is the caller's, so a bad one is
    reported through the not-found path rather than raised — see `read_article`.
    """
    prefix = BRANCH_PREFIX.get(branch)
    if prefix is None:
        raise ValueError(f"unknown guidance branch: {branch!r}")
    key = name.strip().lower()
    if key in TOP_ALIASES:
        return prefix
    # A caller that sends the published slug rather than the area name means the
    # right thing when the branch agrees (`rules='rules_logic'`), so take it. When
    # it disagrees (`rules='Brief_forms'`) it is a branch confusion, and the
    # honest answer is the not-found notice — not a silent read of the OTHER
    # article of that name, and not a wasted round trip for a slug we can already
    # see is wrong.
    for other, other_prefix in BRANCH_PREFIX.items():
        if key.startswith(f"{other_prefix.lower()}_"):
            if other != branch:
                raise ValueError(f"{name!r} names the {other!r} branch, not {branch!r}")
            key = key[len(other_prefix) + 1 :]
            break
    if not AREA_RE.match(key):
        raise ValueError(f"not an area name: {name!r}")
    return f"{prefix}_{key}"


def article_url(branch: str, name: str) -> str:
    return f"{GUIDANCE_BASE_URL}{article_slug(branch, name)}.md"


def _not_found(branch: str, name: str, detail: str = "") -> str:
    return NOT_FOUND_NOTICE.format(branch=branch, name=name, detail=detail)


def read_article(branch: str, name: str, timeout: float | None = None) -> str:
    """One whole guidance article, fenced and stamped, ready to return as-is.

    Three outcomes, deliberately distinct. A name that cannot be an article, or
    a 404, comes back as TEXT saying so — those are answers, and the assistant
    can act on them in the same turn. Anything else (timeout, connection error,
    5xx) PROPAGATES: a fetch that failed for an unknown reason must not be
    dressed up as "this area has no article", and stale or partial guidance is
    worse than a loud failure.
    """
    timeout = FETCH_TIMEOUT if timeout is None else timeout
    try:
        url = article_url(branch, name)
    except ValueError:
        return _not_found(branch, name)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (https only, fixed host)
            body = resp.read().decode("utf-8")
            landed = resp.geturl()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            # The site answers an unknown slug with a full HTML 404 page, so the
            # status is the only reliable signal here — never the body.
            return _not_found(branch, name)
        raise
    # Sanitizing the name secures the URL we ASK for, not the page we get back.
    # urlopen follows redirects silently, so a moved or misconfigured slug could
    # hand back a different article — the other branch's, even — and it would be
    # framed as this one, complete. And a soft 404 answers 200 with an HTML error
    # page. Neither is a rules article, and passing either off as one is worse
    # than any failure, so both become the not-found answer.
    if landed != url:
        return _not_found(branch, name, f" (the request for {url} was redirected to {landed})")
    if body.lstrip()[:1] == "<":
        return _not_found(branch, name, " (the site answered with a page, not the article)")
    key = name.strip().lower()
    label = "top" if key in TOP_ALIASES else key
    return "\n".join((ARTICLE_NOTICE.format(branch=branch, name=label), "",
                       fenced(branch, label, body)))


def fenced(branch: str, label: str, body: str) -> str:
    """One article between its BEGIN and END lines."""
    fields = {"branch": branch, "name": label, "n": len(body), "rev": guidance_version(body)}
    return "\n".join((ARTICLE_BEGIN.format(**fields), body, ARTICLE_END.format(**fields)))
