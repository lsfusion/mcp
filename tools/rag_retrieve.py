from __future__ import annotations
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
    RAG_VECTOR_STORE_ID,
    SOURCETYPE_DOCUMENTATION,
    SOURCETYPE_DOCUMENTATION_PARADIGM,
    SOURCETYPE_DOCUMENTATION_LANGUAGE,
    SOURCETYPE_DOCUMENTATION_HOWTO,
    SOURCETYPE_DOCUMENTATION_BRIEF,
    SOURCETYPE_DOCUMENTATION_RULES,
    SOURCETYPE_DOC_PARADIGM,
    SOURCETYPE_DOC_LANGUAGE,
    SOURCETYPE_DOC_HOWTO,
    SOURCETYPE_DOC_BRIEF,
    SOURCETYPE_DOC_RULES,
    SOURCETYPE,
    SLUG,
    SECTION_ID,
    HEADING_PATH,
    KEYWORDS,
    TOP_K,
    TYPED_TOP_K,
    TITLE_RERANK_CANDIDATE_K,
    TITLE_MATCH_STOPWORDS,
    TITLE_MATCH_BOOST,
    LOCAL_TITLE_MATCH_BOOST,
    HEADING_SECTION_WEIGHT,
    RETRIEVAL_BACKEND,
    EMBEDDING_MODEL,
    QUERY_LOG_MAX_CHARS,
    ERROR_LOG_MAX_CHARS,
)
from tools.event_log import emit
from tools import local_index


class DocItem(BaseModel):
    """Single retrieved chunk from the RAG knowledge base."""
    id: str = Field(..., description="Stable chunk id; pass it in `exclude_ids` to keep this chunk out of a follow-up retrieve_docs call.")
    source: str = Field(..., description="Chunk origin (e.g. documentation-language, documentation-paradigm).")
    text: str = Field(..., description="Retrieved text snippet.")
    score: float = Field(..., description="Raw vector-store similarity score (higher = closer). The list is ranked by this score plus a bounded bonus for query words the article's title or section headings carry, so it does NOT descend across the whole list — the array order IS the ranking, do not re-sort it by this field.")


class RetrieveDocsOutput(BaseModel):
    """List of retrieved chunks sorted by relevance."""
    docs: List[DocItem] = Field(
        default_factory=list,
        description="Relevant chunks returned from the RAG store."
    )


log = logging.getLogger("rag_retrieve")

client = OpenAI(api_key=OPENAI_API_KEY)


@dataclass
class _Hit:
    """One retrieved chunk plus its stable source identifiers (for logging).

    `id` is the chunk's `section_id` attribute — the stable id exposed in
    `DocItem.id` and accepted back in `exclude_ids`.
    `source` is the combined "documentation-<type>" branch label returned to
    consumers; `file_id`/`filename` are the Vector Store identifiers logged so
    a later triage can compare the served result against the live store
    (MCP-FEEDBACK-PLAN.md, Phase A). They are NOT exposed in the tool response.
    """
    id: str
    source: str
    text: str
    score: float
    file_id: str | None
    filename: str | None
    # How much of the query the chunk's heading path carries — see
    # _title_coverage. Worth a bounded addition to `score`, at most
    # TITLE_MATCH_BOOST; 0.0 changes nothing.
    title_coverage: float = 0.0


def _filters_for_source(source_type: str, exclude_ids: list[str] | None = None) -> dict:
    """Attribute filter for one branch, minus its top guidance article if any
    and minus the explicitly excluded `section_id`s.

    All conditions live in ONE flat `and` (the API takes a flat list; nesting
    ands buys nothing and only makes the logged filter harder to read). With a
    single condition the bare comparison is returned as-is.

    `exclude_ids` is applied SERVER-SIDE on purpose: dropping the ids from the
    top-K afterwards would return fewer chunks than asked for, since the store
    would have spent the quota on the excluded ones.
    """
    filters = [{"type": "eq", "key": SOURCETYPE, "value": source_type}]
    top_slug = GUIDANCE_TOP_SLUGS.get(source_type)
    if top_slug is not None:
        filters.append({"type": "ne", "key": SLUG, "value": top_slug})
    # Empty/None => no filter at all: the API contract does not define `nin []`.
    if exclude_ids:
        filters.append({"type": "nin", "key": SECTION_ID, "value": list(exclude_ids)})
    if len(filters) == 1:
        return filters[0]
    return {"type": "and", "filters": filters}


def _vs_search_for_source(
    query: str,
    source_type: str,
    top_k: int,
    exclude_ids: list[str] | None = None,
) -> List[_Hit]:
    """Search the OpenAI Vector Store for chunks tagged with `sourceType=source_type`.

    `source_type` is the bare manifest value ("language" / "paradigm"). The
    returned `_Hit.source` uses the combined "documentation-<type>" form
    so the output shape matches what legacy consumers (platform `RAGRetrieve`,
    plugin) expect.
    """
    if top_k <= 0:
        return []
    # Ask for candidates, not for the answer: the quota is what comes back, and
    # a short query's own article is often outside it (asked for `navigator`,
    # the store put the navigator article at rank 11 of one branch). Costs a
    # bigger response from the store, not another call.
    resp = client.vector_stores.search(
        vector_store_id=RAG_VECTOR_STORE_ID,
        query=query,
        max_num_results=max(top_k, TITLE_RERANK_CANDIDATE_K),
        filters=_filters_for_source(source_type, exclude_ids),
        rewrite_query=False,
    )
    combined = f"{SOURCETYPE_DOCUMENTATION}-{source_type}"
    hits: List[_Hit] = []
    for hit in resp.data:
        text_parts = [c.text for c in (hit.content or []) if getattr(c, "type", None) == "text"]
        attributes = getattr(hit, "attributes", None) or {}
        section_id = attributes.get(SECTION_ID)
        if not section_id:
            # The ingest pipeline stamps `section_id` on EVERY uploaded file
            # (fill/ingest.py:_section_attributes), so a hit without one is a
            # foreign or stale file in a store we do not exclusively own. Drop
            # that one hit — it cannot be identified or excluded later anyway —
            # and keep serving the rest: one bad row must not fail a search that
            # has already spent five branch queries.
            log.warning(
                "retrieve_docs: dropping vector store hit without a %r attribute "
                "(file_id=%r, filename=%r) — stale or foreign file, re-run ragIngestDocs",
                SECTION_ID, getattr(hit, "file_id", None), getattr(hit, "filename", None),
            )
            continue
        hits.append(_Hit(
            id=str(section_id),
            source=combined,
            text="\n".join(text_parts),
            score=float(hit.score or 0.0),
            file_id=getattr(hit, "file_id", None),
            filename=getattr(hit, "filename", None),
            title_coverage=_title_coverage(
                query,
                str(attributes.get(HEADING_PATH) or ""),
                str(attributes.get(SLUG) or ""),
                str(attributes.get(KEYWORDS) or ""),
            ),
        ))
    # The store's ranking, nudged by how much of the query each heading path
    # carries. Then cut to the quota — the branch returns what it always did,
    # chosen from a wider field.
    hits.sort(key=_rank_key)
    return hits[:top_k]



def _embed_query(query: str) -> "np.ndarray":
    """The query as a unit vector, in the model the snapshot was built with."""
    import numpy as np
    v = client.embeddings.create(model=EMBEDDING_MODEL, input=[query]).data[0].embedding
    arr = np.asarray(v, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        raise ValueError("embeddings returned a zero vector for the query")
    return arr / norm


def _local_search_for_source(query: str, query_vector, source_type: str,
                             top_k: int, exclude_ids: list[str] | None) -> List[_Hit]:
    """One branch of the LOCAL index — the same contract as its vector-store
    twin: the branch's own quota, the top guidance article held back, and the
    caller's `exclude_ids` dropped before the cut rather than after."""
    combined = f"{SOURCETYPE_DOCUMENTATION}-{source_type}"
    top_slug = GUIDANCE_TOP_SLUGS.get(source_type)
    rows = local_index.search(
        query_vector, source_type,
        # A wider field than the quota, for the same reason the store call asks
        # for one: the heading bonus can only reorder what it was given.
        max(top_k, TITLE_RERANK_CANDIDATE_K),
        exclude_ids=set(exclude_ids or ()),
        exclude_slugs={top_slug} if top_slug else None,
    )
    hits = [_Hit(
        id=r["section_id"],
        source=combined,
        text=r["text"],
        score=r["score"],
        # The local index has no vector-store identifiers; the section_id is
        # the identity that matters and it is already in `id`.
        file_id=None,
        filename=None,
        title_coverage=_title_coverage(query, r["heading_path"], r["slug"], r["keywords"]),
    ) for r in rows]
    hits.sort(key=_local_rank_key)
    return hits[:top_k]


def _local_rank_key(hit: "_Hit") -> float:
    """Same rule as `_rank_key`, on the cosine's scale — see
    LOCAL_TITLE_MATCH_BOOST for why the coefficient is not the same number."""
    return -(hit.score + LOCAL_TITLE_MATCH_BOOST * hit.title_coverage)


def _rank_key(hit: "_Hit") -> float:
    """The one ranking rule, applied inside a branch and again across branches.

    The store's own ranking, plus at most `TITLE_MATCH_BOOST` for a query whose
    words the article's heading path carries. Bounded and additive on purpose:
    the bonus reorders near neighbours — it cannot lift an article past a hit
    the store scored more than TITLE_MATCH_BOOST higher — and naming nothing
    leaves the store's order untouched. An unconditional tier for a full match was tried
    and dropped — a one-word query like `operator` fully matches a section name
    in dozens of articles, and a tier lets all of them jump a confident
    semantic hit, which the additive form cannot do."""
    return -(hit.score + TITLE_MATCH_BOOST * hit.title_coverage)


def _words(text: str) -> set[str]:
    """Whole words, NFC-normalized and casefolded, Unicode-aware: a Russian
    query must survive as words rather than vanish and leave a stray English
    identifier behind to match a title by accident. `_`, `-` and punctuation
    separate. NFC first so a decomposed `й` or `é` compares equal to the
    composed form a title happens to be written with."""
    text = unicodedata.normalize("NFC", text)
    return {w for w in re.split(r"[^\w]+", text.replace("_", " ").casefold(), flags=re.UNICODE) if w}


def _title_coverage(query: str, heading_path: str, slug: str, keywords: str = "") -> float:
    """What fraction of the query's meaningful words the article's heading path
    carries — the article's own title counted in full, the section titles under
    it at `HEADING_SECTION_WEIGHT`. 1.0 means the query names this article and
    nothing else; 0.0 means it names nothing here and the store's order stands.

    The denominator is EVERY meaningful word of the query, including words no
    heading carries. That is what keeps a paraphrase honest: asked to "forbid
    saving invalid data", an article whose title merely contains `data` covers
    one word in four, not all of the words it happens to know. Weighting the
    matched words by rarity instead was measured and changed nothing, at the
    price of corpus statistics the server does not otherwise need.

    `keywords` — the article's own frontmatter list — counts as its title.
    That is the only answer to a query that names the right article in the
    wrong words: asked to "forbid saving invalid data", the constraints
    article shares NOT ONE word with the query (it says restricted, violate,
    CHECKED BY), so no amount of matching can find it and no weighting can
    rescue it. A `keywords: validation, forbid` on that article can.

    A query naming an article is where the vector search is at its weakest:
    asked for `navigator` it answered with the navigator article at rank 11 of
    its branch, behind longer articles that merely mention navigators. A
    question phrased in the reader's own words matches no heading, scores 0.0
    here, and is returned exactly as the store ranked it — which is what the
    store is good at."""
    wanted = _words(query) - TITLE_MATCH_STOPWORDS
    if not wanted:
        return 0.0
    # "<article title> > <H2> > <H3>" — the head names the article, the tail
    # details it. The slug is the article's own name too, spelled for a URL.
    head, _, tail = heading_path.partition(" > ")
    title = _words(head) | _words(slug) | _words(keywords)
    # A word in BOTH the title and a section counts once, at the title's full
    # weight — never 1.0 + 0.4. That is what keeps coverage within [0, 1] and
    # the bonus within TITLE_MATCH_BOOST.
    section = _words(tail) - title
    return (len(wanted & title)
            + HEADING_SECTION_WEIGHT * len(wanted & section)) / len(wanted)


ALLOWED_TYPES = (
    SOURCETYPE_DOCUMENTATION_LANGUAGE,
    SOURCETYPE_DOCUMENTATION_PARADIGM,
    SOURCETYPE_DOCUMENTATION_HOWTO,
    SOURCETYPE_DOCUMENTATION_BRIEF,
    SOURCETYPE_DOCUMENTATION_RULES,
)

# The one TOP article of each guidance branch, by slug. `get_guidance` already
# serves these two pages in FULL, so their chunks are always in the assistant's
# context — retrieving them again just spends the per-branch quota on text it
# already has. Only these slugs are excluded; the rest of `brief/` and `rules/`
# holds the detailed per-area articles, which exist precisely to be retrieved
# and are searched like any other branch, `type` given or not.
GUIDANCE_TOP_SLUGS = {
    SOURCETYPE_DOCUMENTATION_BRIEF: "Brief",
    SOURCETYPE_DOCUMENTATION_RULES: "Rules",
}

# Branches searched when `type` is omitted.
DEFAULT_TYPES = ALLOWED_TYPES

# `type` argument → TOP_K / TYPED_TOP_K key
_TYPE_TO_TOP_K = {
    SOURCETYPE_DOCUMENTATION_LANGUAGE: SOURCETYPE_DOC_LANGUAGE,
    SOURCETYPE_DOCUMENTATION_PARADIGM: SOURCETYPE_DOC_PARADIGM,
    SOURCETYPE_DOCUMENTATION_HOWTO: SOURCETYPE_DOC_HOWTO,
    SOURCETYPE_DOCUMENTATION_BRIEF: SOURCETYPE_DOC_BRIEF,
    SOURCETYPE_DOCUMENTATION_RULES: SOURCETYPE_DOC_RULES,
}


def retrieve_docs_tool(
    query: str,
    type: str | None = None,
    exclude_ids: list[str] | None = None,
) -> RetrieveDocsOutput:
    """Retrieve chunks from the OpenAI Vector Store populated by the
    `ragIngestDocs` Jenkins pipeline.

    `type` filters by chunk sourceType (the docs folder):
      * omitted / null — search all five branches (`language`, `paradigm`,
        `how-to`, `brief`, `rules`) with a per-branch quota and merge results
        by score.
      * one of `language` / `paradigm` / `how-to` / `brief` / `rules` — only that
        branch, with a per-branch quota of its own (see TYPED_TOP_K).

    `exclude_ids` drops chunks by `DocItem.id`: pass back the ids already in
    context to page deeper into the store instead of getting the same chunks
    again. The exclusion is applied by the store itself, so the response is
    still filled up to the quota.

    The top article of each guidance branch (`Brief`, `Rules`) is never
    returned — `get_guidance` already delivers it in full (see
    GUIDANCE_TOP_SLUGS).

    The store only holds English (`docs/en/`) content. Cross-lingual
    embeddings make non-English queries work, but English wording is
    preferred for best recall.
    """
    start = time.monotonic()
    requested: tuple[str, ...] | None = None
    n_requested: int | None = None
    try:
        if type is not None and type not in ALLOWED_TYPES:
            raise ValueError(
                f"type must be one of {ALLOWED_TYPES} or null/omitted, got {type!r}"
            )
        requested = (type,) if type else DEFAULT_TYPES
        # A quota table per mode: the typed search may spend more on one branch
        # than the untyped search can afford to spend on each of five.
        quotas = TYPED_TOP_K if type else TOP_K
        n_requested = sum(quotas.get(_TYPE_TO_TOP_K[s], 0) for s in requested)
        hits: List[_Hit] = []
        # The local index is preferred only when it is BOTH switched on and
        # actually loaded, and only if the query embeds. Anything else — no
        # snapshot, a refused snapshot, an embeddings outage — falls through to
        # the vector store, which is how this server has always worked. The
        # fallback is the whole reason the switch is safe to flip.
        local = None
        if RETRIEVAL_BACKEND == "local" and local_index.get() is not None:
            try:
                local = _embed_query(query)
            except Exception as e:  # noqa: BLE001 — degrade to the store, do not fail the call
                log.warning("local backend: could not embed the query (%s); "
                            "serving this call from the vector store", e)
        backend = "local" if local is not None else "store"
        for source_type in requested:
            top_k = quotas.get(_TYPE_TO_TOP_K[source_type], 0)
            if local is not None:
                hits.extend(_local_search_for_source(
                    query, local, source_type, top_k, exclude_ids))
            else:
                hits.extend(_vs_search_for_source(query, source_type, top_k, exclude_ids))
        # Ranked the same way across branches — sorting on score alone would
        # undo the per-branch promotion the moment two branches are merged.
        hits.sort(key=_local_rank_key if local is not None else _rank_key)
    except Exception as e:  # noqa: BLE001 — log the failed call, then re-raise unchanged
        _log_retrieval(query, type, n_requested, [], start,
                       ok=False, error=e, exclude_ids=exclude_ids,
                       backend=locals().get("backend", "store"))  # may not be set yet if we failed early
        raise
    # Success path — log OUTSIDE the business `try` so a logging failure can never
    # land in the `except` (false ok=false) or turn a good retrieval into an error.
    # `_log_retrieval` is itself fully guarded and never raises.
    _log_retrieval(query, type, n_requested, hits, start, ok=True,
                   exclude_ids=exclude_ids, backend=backend)
    return RetrieveDocsOutput(
        docs=[DocItem(id=h.id, source=h.source, text=h.text, score=h.score) for h in hits]
    )


def _redact_ids(message: str, ids: list[str] | None) -> str:
    """Replace any caller-supplied chunk id occurring in `message`. Longest
    first, so an id that contains another is not left half-substituted."""
    for i in sorted(ids or [], key=len, reverse=True):
        if i:
            message = message.replace(i, "<excluded-id>")
    return message


def _log_retrieval(
    query: str,
    type_arg: str | None,
    n_requested: int | None,
    hits: List[_Hit],
    start: float,
    *,
    ok: bool,
    backend: str = "store",
    error: BaseException | None = None,
    exclude_ids: list[str] | None = None,
) -> None:
    """Build and emit one `retrieve_docs` event (no chunk text). NEVER raises —
    logging must not break `retrieve_docs` even if field-building itself fails."""
    try:
        fields: dict = {
            "query": (query or "")[:QUERY_LOG_MAX_CHARS],
            "type": type_arg,
            # Chunk BUDGET for this call: the sum of the quota table the call
            # used, across the branches it set out to search — NOT the branch
            # count (e.g. 15 when type is omitted), and NOT a count of searches
            # that completed. On the error path some of those searches may never
            # have run.
            "n_requested": n_requested,
            # Count only: the ids are caller-supplied chunk ids, and the log
            # keeps no chunk identity the caller did not get from us anyway.
            "n_excluded": len(exclude_ids or []),
            "n_results": len(hits),
            # Which backend served this call — the canary needs to be able to
            # tell the two apart in the same log stream.
            "backend": backend,
            # The MAX store score, not the first hit's: the list is ranked by
            # score plus the title bonus, so the head of it need not be the
            # best-scoring chunk. Gap analytics compare stores, not orders.
            "top_score": (max(h.score for h in hits) if hits else None),
            "latency_ms": int((time.monotonic() - start) * 1000),
            "results": [
                {
                    "rank": i + 1,
                    "source": h.source,
                    "file_id": h.file_id,
                    "filename": h.filename,
                    "score": h.score,
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
