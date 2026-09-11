from __future__ import annotations
import collections
import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import List
from pydantic import BaseModel, Field

from openai import OpenAI

from settings import (
    OPENAI_API_KEY,
    SOURCETYPE_DOCUMENTATION,
    SOURCETYPE_DOCUMENTATION_PARADIGM,
    SOURCETYPE_DOCUMENTATION_LANGUAGE,
    SOURCETYPE_DOCUMENTATION_HOWTO,
    SOURCETYPE_DOCUMENTATION_BRIEF,
    SOURCETYPE_DOCUMENTATION_RULES,
    SOURCETYPE,
    RESULT_MAX_CHARS,
    SCORE_FLOOR_GAP,
    EMBEDDING_MODEL,
    ARTICLE_MAX_NAMES,
    BATCH_MAX_QUERIES,
    QUERY_LOG_MAX_CHARS,
    ERROR_LOG_MAX_CHARS,
)
from tools.caller import caller_fields
from tools.guidance import DOC_ID_SEP
from tools.event_log import emit
from tools import local_index


class DocItem(BaseModel):
    """Single retrieved chunk from the RAG knowledge base."""
    id: str = Field(..., description="Stable chunk id, shaped `<article>::<section>`. Two uses, and the second is the one worth knowing: pass it in `exclude_ids` to keep this chunk out of a follow-up call, or pass it as `article` — unchanged, section and all — to read the WHOLE article this chunk came from. A chunk cannot show you the constraint it depends on, the case it omits, or the table it points at; those sit elsewhere in the same article. Reaching for the article is the cheap move there. Searching again with different wording is the expensive one.")
    source: str = Field(..., description="Chunk origin (e.g. documentation-language, documentation-paradigm).")
    text: str = Field(..., description="Retrieved text snippet.")
    score: float | None = Field(
        default=None,
        description="Similarity to the query, higher = closer. NULL when no query was given — an article traversal has nothing to be similar to, and inventing a number there would make the ordering look like a ranking. When it is null the list is in document order; when it is set the list is ranked by it, descending.")
    query: str | None = Field(
        default=None,
        description="Which of the submitted queries this chunk answers. Null when only one was submitted.")


class RetrieveDocsOutput(BaseModel):
    """List of retrieved chunks sorted by relevance."""
    docs: List[DocItem] = Field(
        default_factory=list,
        description="Relevant chunks returned from the RAG store."
    )
    status: str = Field(
        default="",
        description="Whether this response is the whole answer, and what to do when it is not. Always present, always a sentence, because the decision it drives — page again, or stop — is one a number leaves the reader to infer. It opens with a fixed label: COMPLETE, MORE, ARTICLE_COMPLETE, ARTICLE_PARTIAL or ARTICLE_NONE. Counts stay inside it; the log keeps them separately for us.")


log = logging.getLogger("rag_retrieve")

client = OpenAI(api_key=OPENAI_API_KEY)


@dataclass
class _Hit:
    """One retrieved chunk plus its stable source identifiers (for logging).

    `id` is the chunk's `section_id` attribute — the stable id exposed in
    `DocItem.id` and accepted back in `exclude_ids`.
    `source` is the combined "documentation-<type>" branch label returned to
    consumers; the section id in `id` is the only identity a chunk has, so
    a later triage can compare the served result against the live store
    (MCP-FEEDBACK-PLAN.md, Phase A). They are NOT exposed in the tool response.
    """
    id: str
    source: str
    text: str
    score: float
    # Which of the submitted queries this chunk answers; None for a single one.
    query: str | None = None
    # Position in its own article. Internal: it is how `_status` says which
    # part of the article this response holds. It is NOT a response field —
    # pages of a traversal are contiguous runs, so list order is reading
    # order, and the status names the positions in words.
    ordinal: int | None = None



def _embed_queries(queries: list[str]) -> list["np.ndarray"]:
    """The queries as unit vectors, in the model the snapshot was built with.

    ONE request for the whole batch: this round trip is ~95% of a call's time
    (193 ms of ~205), and it takes a list at the same price — four queries come
    back in 182 ms, where four separate calls take 728.
    """
    import numpy as np
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=queries)
    out: list[np.ndarray] = [None] * len(queries)  # type: ignore[list-item]
    # Ordered by the item's own index, not by position: the API returns them in
    # order today, and pairing a query with the wrong vector would not fail.
    for item in resp.data:
        arr = np.asarray(item.embedding, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm == 0.0:
            raise ValueError(f"embeddings returned a zero vector for query {item.index}")
        out[item.index] = arr / norm
    if any(v is None for v in out):
        raise ValueError("embeddings returned fewer vectors than queries")
    return out


def _hit_from_row(row: dict, query: str | None) -> _Hit:
    """One local-index row as a `_Hit`, carrying its place in its article."""
    return _Hit(
        id=row["section_id"],
        source=f"{SOURCETYPE_DOCUMENTATION}-{row['sourceType']}",
        text=row["text"],
        score=row["score"],
        query=query,
        ordinal=row.get("ordinal"),
    )


def _search_branch(query_vector, source_type: str,
                   exclude_ids: list[str] | None) -> List[_Hit]:
    """Candidates from ONE branch, best first, deep enough to fill the budget."""
    combined = f"{SOURCETYPE_DOCUMENTATION}-{source_type}"
    rows = local_index.search(
        query_vector, source_type, RESULT_MAX_CHARS,
        exclude_ids=set(exclude_ids or ()),
    )
    hits = [_Hit(
        id=r["section_id"],
        source=combined,
        text=r["text"],
        score=r["score"],
    ) for r in rows]
    return hits


# The searchable corpus. `brief` and `rules` are NOT here: relevance in these
# three branches is probabilistic, so ranked excerpts are the right answer,
# whereas a rules article is only useful whole and a top-N retrieval cannot
# report what it withheld — an assistant handed 3 of an article's 4 chunks has
# no way to learn a 4th existed. Those two are read by name, entire, through
# `get_guidance`, and asking for them here is now an error naming the three
# that remain.
ALLOWED_TYPES = (
    SOURCETYPE_DOCUMENTATION_LANGUAGE,
    SOURCETYPE_DOCUMENTATION_PARADIGM,
    SOURCETYPE_DOCUMENTATION_HOWTO,
)

# Branches searched when `type` is omitted.
DEFAULT_TYPES = ALLOWED_TYPES

# The two guidance branches are not in this index, and a caller can arrive at
# their names honestly: the corpus links to `Brief.md` and `Rules.md`, and those
# are the only two of its 3,038 `.md` destinations that do not name a reference
# article. Answering those with an empty result would teach exactly the wrong
# lesson — that the subject is undocumented — so they are routed instead.
_GUIDANCE_PREFIXES = ("brief", "rules")


def resolve_article(name: str) -> str:
    """The article a caller meant, or raise ValueError saying what to do instead.

    Three shapes are accepted, because those are the three a caller can honestly
    be holding and none of them is worth making them reformat:

      * a bare slug — `Interactive_view`;
      * a chunk id — `Interactive_view::examples`; everything from `::` on names
        the section, and the article is what gets read;
      * the DESTINATION of a documentation link — `Actions.md` or
        `../paradigm/Actions.md`, the only two shapes the corpus uses. The
        destination, never the visible label: a label is prose.

    The rule for a link is the whole basename minus `.md`. Measured over the
    corpus it yields a real article for 3,036 of 3,038 links; the two it does
    not are the guidance top articles, which get routed.
    """
    raw = (name or "").strip()
    slug = raw.split(DOC_ID_SEP, 1)[0].split("#", 1)[0].strip()
    slug = slug.rsplit("/", 1)[-1]
    if slug.endswith(".md"):
        slug = slug[:-3]
    if not slug:
        raise ValueError(
            f"{name!r} does not name an article. An article name is a published "
            "slug (`Interactive_view`); a chunk `id` up to `::` is one, and so is "
            "the basename of any `.md` link in the documentation.")
    if slug.lower().startswith(_GUIDANCE_PREFIXES):
        raise ValueError(
            f"{slug!r} is a guidance article, and guidance is not searched. Read "
            f"it whole with `lsfusion_get_guidance` instead: the top article of "
            f"each branch comes back from a call with NO arguments, and an area's "
            f"article from `rules='<area>'` or `brief='<area>'`. NOTHING WAS READ "
            f"here, and nothing about the subject follows from this error.")
    if not local_index.known_article(slug):
        raise ValueError(
            f"no article named {slug!r}. This is NOT a finding that the subject is "
            f"undocumented — nothing was searched. Article names come from two "
            f"places, both already in front of you: the part of any chunk `id` "
            f"before `::`, and the basename of any `.md` link inside chunk text "
            f"(`../paradigm/Actions.md` means `Actions`). Search with `query` and "
            f"take the name off a chunk that looks close.")
    return slug


def retrieve_docs_tool(
    query: str | list[str] | None = None,
    type: str | None = None,
    exclude_ids: list[str] | None = None,
    article: str | None = None,
) -> RetrieveDocsOutput:
    """Retrieve chunks for one information need, or for several at once.

    `query` may be a single string, or a LIST when the caller already knows it
    needs several unrelated things — events AND the navigator, say. On real
    traffic that is the normal case rather than the exception: 60% of
    consecutive calls are on unrelated topics, and 85% of call bursts carry two
    or more. Batching them is worth doing because the cost of a call is almost
    entirely the one network round trip that embeds the query, and that request
    takes a list: two queries cost 194 ms instead of 581, four cost 182 instead
    of 728.

    A batch shares ONE budget (`BATCH_TOTAL_CAP`) rather than multiplying it, so
    it never costs the caller MORE context than the separate calls it replaces,
    and a chunk that answers two of the queries is returned once — credited to
    the query it scored higher for. Each returned chunk says which query that
    was. `per_query` is a ceiling, not a guaranteed share: a query whose best
    chunks all belong to a neighbour returns fewer, which is the correct answer
    to asking two versions of the same thing.

    What a batch gives up is the loop: asking one thing, reading the answer, and
    letting it shape the next question. Batch what is known up front; keep
    asking one at a time when each answer changes the next question.

    `type` filters by chunk sourceType (the docs folder) and applies to every
    query in the batch:
      * omitted / null — search all three reference branches (`language`,
        `paradigm`, `how-to`) with a per-branch quota and merge results by score.
      * one of `language` / `paradigm` / `how-to` — only that branch.

    The `brief` and `rules` branches are not searchable and are rejected here:
    an area's capability map and its coding rules are read whole, by name, with
    `get_guidance`.

    `exclude_ids` drops chunks by `DocItem.id` and likewise applies to the whole
    batch: pass back the ids already in context to page deeper instead of
    getting the same chunks again.

    The store only holds English (`docs/en/`) content. Cross-lingual
    embeddings make non-English queries work, but English wording is
    preferred for best recall.
    """
    start = time.monotonic()
    if query is None and article is None:
        raise ValueError(
            "give `query`, `article`, or both. `query` searches by meaning; "
            "`article` names one article and walks it in document order; "
            "together they search inside that one article.")
    queries = [] if query is None else ([query] if isinstance(query, str) else list(query))
    # Identical strings are collapsed BEFORE the budget is divided: otherwise
    # ["x", "x"] halves the share and then hands every chunk to the first copy,
    # leaving the second with nothing and the caller with less than a plain "x"
    # would have returned. Order is preserved.
    queries = list(dict.fromkeys(queries)) if len(queries) > 1 else queries
    requested: tuple[str, ...] | None = None
    n_candidates: int | None = None
    article_slugs: list[str] = []
    unresolved_names: list[str] = []
    # slug -> (chunks in the whole article, chunks this call could still offer
    # after `exclude_ids`). The first is the denominator, the second is what
    # says whether anything is left to fetch.
    article_totals: dict[str, tuple[int, int]] = {}
    try:
        if any(not isinstance(q, str) or not q.strip() for q in queries):
            raise ValueError("query must be a non-empty string, or a list of them")
        if not queries and article is None:
            raise ValueError("query must be a non-empty string, or a list of them")
        if len(queries) > BATCH_MAX_QUERIES:
            raise ValueError(
                f"a batch carries at most {BATCH_MAX_QUERIES} queries, got {len(queries)}. "
                f"Past that this call cannot both answer every query and stay inside "
                f"one response, and answering some of them with silence would look "
                f"exactly like finding nothing. Split it: a batch costs the same as "
                f"the separate calls it replaces, so nothing is lost by doing so.")
        if type is not None and type not in ALLOWED_TYPES:
            raise ValueError(
                f"type must be one of {ALLOWED_TYPES} or null/omitted, got {type!r}"
            )
        # An article names its own branch, so `type` is redundant with it and a
        # `type` that disagrees is a contradiction rather than a narrowing.
        # Slugs are globally unique, which is what makes the branch derivable.
        if article is not None:
            names = [article] if isinstance(article, str) else list(article)
            names = list(dict.fromkeys(n for n in names if str(n).strip()))
            if not names:
                raise ValueError("article must be a non-empty name, or a list of them")
            if len(names) > ARTICLE_MAX_NAMES:
                raise ValueError(
                    f"a call reads at most {ARTICLE_MAX_NAMES} articles, got "
                    f"{len(names)}. Past that this call cannot both give every "
                    f"article a foothold and stay inside one response, and an "
                    f"article returned empty would look exactly like an article "
                    f"that does not exist. Split it.")
            # A name that resolves to nothing is REPORTED when it travels with
            # names that do resolve: the readable articles are still readable,
            # and raising would throw them away over someone else's typo. When
            # NOTHING resolves there is no response to attach a report to, so
            # the first failure is raised with its own recovery instructions.
            article_slugs, unresolved = [], []
            for n in names:
                try:
                    article_slugs.append(resolve_article(n))
                except ValueError as exc:
                    unresolved.append((n, exc))
            if not article_slugs:
                raise unresolved[0][1]
            unresolved_names = [n for n, _ in unresolved]
            branches = {local_index.article_branch(sl) for sl in article_slugs}
            branches.discard(None)
            if type is not None and branches and type not in branches:
                raise ValueError(
                    f"the named article(s) are in {sorted(branches)!r}, but "
                    f"type={type!r} was given. Drop `type`: naming an article "
                    f"already picks its branch.")
            requested: tuple[str, ...] = tuple(sorted(branches)) or DEFAULT_TYPES
        else:
            requested = (type,) if type else DEFAULT_TYPES

        # DOCUMENT order when an article was named with no query: there is
        # nothing to rank by, and the order the article reads is the answer.
        document_order = bool(article_slugs) and not queries

        # One cell per (query, branch), or the single cell of a traversal. The
        # character budget is split evenly between them and unused share flows
        # on, which is what keeps a strong branch — or one query of a batch —
        # from taking the whole response.
        n_cells = (len(article_slugs) * max(1, len(queries)) if article_slugs
                   else len(queries) * len(requested))
        cell_share = RESULT_MAX_CHARS // n_cells

        # There is one backend. A snapshot that will not load, or a query that
        # will not embed, is an ERROR the caller sees — never an empty result
        # set, which to a model is indistinguishable from "no documentation
        # exists on this subject" and arrives labelled a success.
        vectors: list | None = None
        if local_index.get() is None:
            raise RuntimeError(
                "documentation retrieval is unavailable: the local index did "
                "not load. This is NOT a finding that no documentation matches "
                "— nothing was searched. Do not conclude anything about the "
                "subject from this error; report it and continue.")
        vectors = _embed_queries(queries) if queries else []

        if article_slugs:
            # Walking articles is not a search: the candidate set is the
            # articles themselves, and the only question is the order. A total
            # counts the WHOLE article and ignores `exclude_ids`, so it stays
            # the denominator across pages.
            #
            # Cells are per article in request order — the caller's order, not
            # ours. It is the only one they can predict, and predictability is
            # what lets them read a continuation and know what it will do.
            cells: list[List[_Hit]] = []
            for i in range(max(1, len(queries))):
                qv = vectors[i] if queries else None
                for slug in article_slugs:
                    rows, total = local_index.article_chunks(
                        slug, qv, set(exclude_ids or ()))
                    article_totals[slug] = (total, len(rows))
                    cells.append([_hit_from_row(r, queries[i] if len(queries) > 1 else None)
                                  for r in rows])
        else:
            # Every query searched first, then the shared budget handed out. Two
            # passes rather than one because a chunk that answers two of the
            # queries belongs to the one that ranked it higher, and that is not
            # knowable while the first query is still being served.
            found: list[dict[str, List[_Hit]]] = []
            for i, q in enumerate(queries):
                per_branch: dict[str, List[_Hit]] = {}
                for source_type in requested:
                    branch_hits = _search_branch(vectors[i], source_type, exclude_ids)
                    # Ranked within its own branch. Kept SEPARATE rather than
                    # merged here: the budget below hands out a share per branch,
                    # and that is only possible while it still knows which branch
                    # each hit came from.
                    branch_hits.sort(key=lambda h: -h.score)
                    per_branch[source_type] = branch_hits
                found.append(per_branch)

            # Who owns each chunk: the query it scored highest FOR. A chunk is
            # returned ONCE — the caller's context should not hold it twice, and
            # separate calls could never know it was a repeat.
            owner: dict[str, int] = {}
            best: dict[str, float] = {}
            for i, per_branch in enumerate(found):
                for lst in per_branch.values():
                    for h in lst:
                        if h.id not in best or h.score > best[h.id]:
                            best[h.id], owner[h.id] = h.score, i
            cells = []
            for i, per_branch in enumerate(found):
                for source_type in requested:
                    cell = [h for h in per_branch.get(source_type, ())
                            if owner.get(h.id) == i]  # another query answers it better
                    for h in cell:
                        h.query = queries[i] if len(queries) > 1 else None
                    cells.append(cell)
        # Relevance floor, applied ACROSS branches and before the budget so the
        # room freed goes to nothing rather than to noise. Measured DOWN from
        # the best score this query found, because "good" only means anything
        # next to what else was found: a fixed threshold would answer a hard
        # query with nothing and an easy one with everything. A gap rather than
        # a fraction, because a fraction of a low best score lands in the noise
        # — see SCORE_FLOOR_GAP for the numbers that settled it.
        #
        # Across BRANCHES but within ONE query. A branch whose own best is below
        # the floor is not answering the question, and dropping it whole is the
        # honest form of that — its share then flows to the branches that are.
        # Across queries it would be wrong: two unrelated questions have
        # unrelated score scales, and a strong match for one would silently
        # delete the only answer the other had.
        if not document_order:
            per_q = len(requested)
            for q0 in range(0, len(cells), per_q):
                group = cells[q0:q0 + per_q]
                best_score = max((h.score for cell in group for h in cell), default=None)
                if best_score is None or best_score <= 0:
                    continue
                floor = best_score - SCORE_FLOOR_GAP
                cells[q0:q0 + per_q] = [[h for h in cell if h.score >= floor]
                                        for cell in group]
        n_candidates = sum(len(c) for c in cells)
        revision = local_index.revision()

        # Two passes, and the second is what makes the first safe to be strict.
        #
        # Pass one gives every cell the same share, so a strong branch cannot
        # crowd out the other two and one query of a batch cannot take the whole
        # response — the guarantee the old per-branch chunk quota used to
        # provide, now expressed in the unit that actually costs the caller
        # something. Pass two spends whatever the cells did not, best first
        # across all of them, so an even split never wastes room: a cell with
        # nothing good in it releases its share instead of holding it.
        #
        # Chunks are kept WHOLE. Half a chunk answers nothing and cannot be
        # paged for, so the budget drops chunks rather than trimming tails.
        #
        # In DOCUMENT order the walk STOPS at the first chunk that does not fit,
        # and there is no second pass. Skipping ahead would hand back sections
        # 1, 2 and 4 as if they were consecutive, with nothing in the response
        # saying otherwise — a hole a reader cannot see is worse than a short
        # answer it can page for. What comes back is a prefix of the chunks
        # still eligible after `exclude_ids`.
        hits: List[_Hit] = []
        picked: set[str] = set()
        used = 0
        if document_order:
            # Two rules, and the order of them is the design.
            #
            # First a FOOTHOLD: every article gets its next chunk, whatever its
            # share, because an article returned empty is indistinguishable from
            # an article that does not exist — the same reason a query is never
            # left with nothing. Reserved before anything else, so a big article
            # earlier in the list cannot eat the foothold of a small one later.
            #
            # Then work-conserving redistribution: keep going round, in request
            # order, taking each article's next chunk while it fits, until a
            # full pass adds nothing. A single second pass would strand budget
            # that a later article released.
            for cell in cells:
                if cell and used + len(cell[0].text) <= RESULT_MAX_CHARS:
                    hits.append(cell[0])
                    picked.add(cell[0].id)
                    used += len(cell[0].text)
            taken = {id(cell): (1 if cell and cell[0].id in picked else 0) for cell in cells}
            while True:
                progress = False
                for cell in cells:
                    k = taken[id(cell)]
                    if k >= len(cell):
                        continue
                    nxt = cell[k]
                    if used + len(nxt.text) > RESULT_MAX_CHARS:
                        continue
                    hits.append(nxt)
                    picked.add(nxt.id)
                    used += len(nxt.text)
                    taken[id(cell)] = k + 1
                    progress = True
                if not progress:
                    break
        else:
            for cell in cells:
                spent = 0
                for h in cell:
                    if spent + len(h.text) > cell_share:
                        continue
                    hits.append(h)
                    picked.add(h.id)
                    spent += len(h.text)
                    used += len(h.text)
            # No query leaves with nothing while it had something to give. An
            # even split stops being a fair split once a share is smaller than
            # a chunk: the cells win or lose whole chunks, so a query whose
            # best chunk is merely bigger than its share is answered by the
            # global pass or not at all — and "not at all" is indistinguishable
            # from "nothing matched". This is the one place the split is
            # allowed to be exceeded, and it is what lets the batch cap be a
            # measured boundary rather than a guess.
            answered = {h.query for h in hits}
            for i, q in enumerate(queries):
                label = q if len(queries) > 1 else None
                if label in answered:
                    continue
                best = min((h for cell in cells[i * len(requested):(i + 1) * len(requested)]
                            for h in cell if h.id not in picked),
                           key=lambda h: -h.score, default=None)
                if best is None:
                    continue
                hits.append(best)
                picked.add(best.id)
                used += len(best.text)
            for h in sorted((h for cell in cells for h in cell if h.id not in picked),
                            key=lambda h: -h.score):
                if hits and used + len(h.text) > RESULT_MAX_CHARS:
                    continue
                hits.append(h)
                picked.add(h.id)
                used += len(h.text)
        n_omitted = n_candidates - len(hits)
        # Handed out per cell, returned as one ranked list: `score` is what
        # DocItem promises the order means, and ownership already compared
        # scores across queries, so they are comparable here too. A traversal
        # has no scores and keeps the order it was built in.
        if not document_order:
            hits.sort(key=lambda h: -h.score)
    except Exception as e:  # noqa: BLE001 — log the failed call, then re-raise unchanged
        _log_retrieval(queries, type, n_candidates, [], start,
                       ok=False, error=e, exclude_ids=exclude_ids)
        raise
    # Success path — log OUTSIDE the business `try` so a logging failure can never
    # land in the `except` (false ok=false) or turn a good retrieval into an error.
    # `_log_retrieval` is itself fully guarded and never raises.
    _log_retrieval(queries, type, n_candidates, hits, start, ok=True,
                   exclude_ids=exclude_ids, n_omitted=n_omitted,
                   revision=revision,
                   article=",".join(article_slugs) or None)
    return RetrieveDocsOutput(
        docs=[DocItem(id=h.id, source=h.source, text=h.text, score=h.score,
                      query=h.query)
              for h in hits],
        status=_status(article_slugs, article_totals, hits, n_omitted,
                       n_candidates, revision, searched=bool(queries),
                       unresolved=unresolved_names),
        omitted=n_omitted,
    )


def _ordinal_ranges(ordinals: list[int]) -> str:
    """`0-8`, or `0-3,7,9-11` when exclusions have left holes.

    A bare min-max would read as coverage, and after `exclude_ids` it is not.
    """
    out, i = [], 0
    while i < len(ordinals):
        j = i
        while j + 1 < len(ordinals) and ordinals[j + 1] == ordinals[j] + 1:
            j += 1
        out.append(str(ordinals[i]) if i == j else f"{ordinals[i]}-{ordinals[j]}")
        i = j + 1
    return ",".join(out)


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


# Said where the reader already is, not only where they would look after
# working out that the door exists. The reported failure was an agent that
# could not get from "this chunk is close" to "show me that article" and kept
# rephrasing its query instead — and the `article` parameter's own description,
# which explains the move, is read only by someone who has already decided to
# make it.
_READ_WHOLE = (" If one of them is on the right subject but plainly partial, read its "
               "whole article: pass that chunk's `id` as `article`.")


def _status(slugs: list[str], totals: dict[str, tuple[int, int]], hits: List[_Hit],
            omitted: int, candidates: int, revision: str | None, searched: bool,
            unresolved: list[str] | None = None) -> str:
    """Is this response the whole answer, and if not, what call continues it.

    One field for every mode and always a sentence, because the decision it
    drives — call again, or stop — is one a number leaves the reader to infer.
    A whole-article read proves completeness with a terminator: the END fence
    goes missing along with a truncated tail, so it cannot lie. A JSON array has
    no tail to lose, so nine chunks of nineteen look exactly like all nine there
    are.

    Two things it will not do. It never says what the CALLER holds — the server
    cannot see what earlier pages left in their hands — so "complete" is always
    about this walk and this revision. And it never emits a continuation it
    cannot honour: when nothing is left the sentence is terminal and offers no
    next call, because a continuation that returns the same page forever is
    worse than a full stop.
    """
    more = (f" {_plural(omitted, 'chunk', 'chunks')} did not fit this response's "
            f"size budget." if omitted else "")
    page_on = (" To continue, repeat this call with the ids above added to "
               "`exclude_ids`, keeping every id you have already excluded."
               if omitted else "")

    if not slugs:
        if not omitted:
            return ("COMPLETE: every chunk that cleared the relevance floor for this "
                    "query is in this response. Paging will not lengthen it." + _READ_WHOLE)
        return f"MORE:{more}{page_on}{_READ_WHOLE}"

    # Searching inside a set of articles is a SEARCH — it is scoped, not a walk,
    # and it must never claim the articles were delivered.
    if searched:
        where = _names(slugs)
        if not omitted:
            return (f"COMPLETE: every chunk of {where} that cleared the relevance "
                    f"floor is in this response. This is a search inside them, not "
                    f"the articles themselves — to read one whole, name it with no "
                    f"query.")
        return (f"MORE:{more}{page_on} This is a search inside {where}, not the "
                f"articles themselves.")

    got = collections.Counter(h.id.split(DOC_ID_SEP, 1)[0] for h in hits)
    unknown = list(unresolved or []) + [sl for sl in slugs
                                        if not totals.get(sl, (0, 0))[0]]
    known = [sl for sl in slugs if totals.get(sl, (0, 0))[0]]
    # Exhausted means nothing of it remains to fetch — everything this call did
    # not exclude came back. It is about the WALK, which is why an article
    # bigger than one response can still reach it, over pages.
    left = {sl: candidates_left for sl, candidates_left in
            ((sl, _left_of(sl, hits, totals)) for sl in known)}
    unfinished = [sl for sl in known if left[sl] and got.get(sl)]
    stuck = [sl for sl in known if left[sl] and not got.get(sl)]
    done = [sl for sl in known if not left[sl]]

    note = ""
    if stuck:
        note += (f" {_names(stuck)} could not be started in this response; if a "
                 f"continuation carrying fewer articles still returns nothing of "
                 f"{'it' if len(stuck) == 1 else 'them'}, its next chunk is larger "
                 f"than one whole response and no further call will reach it.")
    if unknown:
        note = (f" No article is named {_names(unknown)}; that name resolves to "
                f"nothing and will not resolve on a retry — search for it "
                f"instead.")
    at = f"at revision {revision}"

    def count(sl: str, finished: bool) -> str:
        """What this response holds of one article.

        A fraction beside the word COMPLETE reads as a contradiction — "9 of
        19" and "nothing remains" are both true on the last page of a walk, and
        together they look like a mistake. So a finished article says how much
        arrived HERE, and an unfinished one says how much of the whole.
        """
        total = totals.get(sl, (0, 0))[0]
        n = got.get(sl, 0)
        if n == total:
            return f"`{sl}` all {_plural(total, 'chunk', 'chunks')}"
        if finished:
            return f"`{sl}` {_plural(n, 'chunk', 'chunks')} here, none left"
        return f"`{sl}` {n} of {total}"

    # Two ways to come back empty, and calling one the other would be a lie.
    # Excluded means the caller already holds it. Too big means the next chunk
    # does not fit a whole response, and no continuation will ever change that
    # — so it is named as a dead end rather than offered as a retry.
    too_big = [sl for sl in known if totals[sl][1] and not got.get(sl)]
    spent = [sl for sl in known if not totals[sl][1]]
    if not got:
        why = ""
        if spent:
            why += (f" Every chunk of {_names(spent)} was already removed by "
                    f"`exclude_ids`.")
        if too_big:
            why += (f" {_names(too_big)} could not be started: the next chunk of "
                    f"{'it' if len(too_big) == 1 else 'them'} is larger than one "
                    f"whole response, so no continuation will reach it.")
        return f"ARTICLE_NONE: no chunks came back {at}.{why}{note}"
    if not unfinished:
        return (f"ARTICLE_COMPLETE: {_join(count(sl, True) for sl in done)} {at} — "
                f"nothing of {'it' if len(done) == 1 else 'them'} remains to "
                f"fetch.{note}")
    head = (f"ARTICLE_PARTIAL: {_join(count(sl, True) for sl in done) + ' complete; ' if done else ''}"
            f"{_join(count(sl, False) + f' ({left[sl]} still to fetch)' for sl in unfinished)}, "
            f"{at}.{more}{note}")
    return (f"{head} To continue, repeat with `article={unfinished!r}` and the ids "
            f"above added to `exclude_ids`, keeping every id you have already "
            f"excluded; a revision that changes under you means the walk starts "
            f"again without the old exclusions.")


def _left_of(slug: str, hits: List[_Hit], totals: dict[str, tuple[int, int]]) -> int:
    """Chunks of this article still fetchable after this response.

    The rows the index offered already had `exclude_ids` removed, so what is
    left is those rows minus what actually went out. This is why an article
    larger than one response can still reach "nothing remains" — over pages.
    """
    _total, offered = totals.get(slug, (0, 0))
    returned = sum(1 for h in hits if h.id.split(DOC_ID_SEP, 1)[0] == slug)
    return max(0, offered - returned)


def _names(slugs) -> str:
    return _join(f"`{s}`" for s in slugs)


def _join(parts) -> str:
    q = list(parts)
    if not q:
        return ""
    if len(q) == 1:
        return q[0]
    return ", ".join(q[:-1]) + " and " + q[-1]


def _cap_total(queries: list[str], budget: int) -> list[str]:
    """The queries, trimmed so the WHOLE list fits the budget. Capping each one
    separately would let a batch write the cap times the batch size."""
    out: list[str] = []
    left = budget
    for q in queries:
        if left <= 0:
            break
        out.append(q[:left])
        left -= len(out[-1])
    return out


def _redact_ids(message: str, ids: list[str] | None) -> str:
    """Replace any caller-supplied chunk id occurring in `message`. Longest
    first, so an id that contains another is not left half-substituted."""
    for i in sorted(ids or [], key=len, reverse=True):
        if i:
            message = message.replace(i, "<excluded-id>")
    return message


def _log_retrieval(
    queries: list[str],
    type_arg: str | None,
    n_candidates: int | None,
    hits: List[_Hit],
    start: float,
    *,
    ok: bool,
    error: BaseException | None = None,
    exclude_ids: list[str] | None = None,
    n_omitted: int = 0,
    revision: str | None = None,
    article: str | None = None,
) -> None:
    """Build and emit one `retrieve_docs` event (no chunk text). NEVER raises —
    logging must not break `retrieve_docs` even if field-building itself fails."""
    try:
        fields: dict = {
            # `query` stays a single string whatever was asked, so every
            # existing reader keeps working — but for a batch it is a JOIN, not
            # something anyone typed, and analytics that group by it would
            # invent a topic. `queries` carries the real ones, and analytics
            # should group by those. Capped as a whole, not per item: eight
            # queries at the per-query cap would otherwise write 16 KB.
            "query": (" | ".join(queries))[:QUERY_LOG_MAX_CHARS],
            **({"queries": _cap_total(queries, QUERY_LOG_MAX_CHARS)}
               if len(queries) > 1 else {}),
            # Per query, because one strong query hides another that returned
            # nothing: the aggregate n_results and top_score cannot show that a
            # topic failed.
            **({"query_stats": [
                {"query_index": i,
                 "n_results": sum(1 for h in hits if h.query == q),
                 "top_score": max((h.score for h in hits if h.query == q), default=None)}
                for i, q in enumerate(queries)]}
               if len(queries) > 1 else {}),
            "type": type_arg,
            # Candidates the branches actually returned, BEFORE the character
            # budget chose among them. With `n_results` and `n_omitted` beside
            # it this says whether the budget or the corpus decided the answer —
            # which is the pair needed to tell whether the budget is set
            # anywhere near right. Null on the error path, where some searches
            # may never have run.
            "n_candidates": n_candidates,
            # Count only: the ids are caller-supplied chunk ids, and the log
            # keeps no chunk identity the caller did not get from us anyway.
            "n_excluded": len(exclude_ids or []),
            "n_results": len(hits),
            # How often the SIZE budget, not the chunk quota, decided the
            # answer. Without this the two look identical in the log — a short
            # result reads as "little matched" either way — and there would be
            # no way to tell whether the budget is set anywhere near right.
            "n_omitted": n_omitted,
            # The corpus generation that answered. Never in the RESPONSE — the
            # only caller-facing use for it is a traversal spanning a rebuild,
            # and `article_status` says it there in words. Here it is the field
            # that lets a bad answer be pinned to the snapshot that gave it,
            # which is a question only we ever ask.
            "revision": revision,
            # Set when the call NAMED an article instead of searching. The two
            # are different enough that mixing them in one rate would hide
            # both.
            "article": article,
            # The MAX store score, not the first hit's: the list is ranked by
            # score plus the title bonus, so the head of it need not be the
            # best-scoring chunk. Gap analytics compare stores, not orders.
            "top_score": (max(h.score for h in hits) if hits else None),
            "latency_ms": int((time.monotonic() - start) * 1000),
            # Which client, never which model — see tools/caller.py.
            **caller_fields(),
            "results": [
                {
                    "rank": i + 1,
                    "source": h.source,
                    "score": h.score,
                    # Which query this one answers; absent for a single query.
                    **({"query_index": queries.index(h.query)} if h.query else {}),
                }
                for i, h in enumerate(hits)
            ],
        }
        if not ok and error is not None:
            fields["error_class"] = type(error).__name__
            # A provider error can echo the request back, filter values and all,
            # which would smuggle the caller's excluded ids into the log through
            # a field that never names them. Scrub them before truncating.
            fields["error_message"] = _redact_ids(str(error), exclude_ids)[:ERROR_LOG_MAX_CHARS]
        emit("retrieve_docs", fields, stream="retrieval", ok=ok)
    except Exception:  # noqa: BLE001 — best-effort; never propagate into retrieve_docs
        pass
