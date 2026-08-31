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
    TOP_K,
    RETRIEVAL_BACKEND,
    EMBEDDING_MODEL,
    BATCH_TOTAL_CAP,
    BATCH_MAX_QUERIES,
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
    score: float = Field(..., description="Similarity score, higher = closer. The list is ranked by it.")
    query: str | None = Field(
        default=None,
        description="Which of the submitted queries this chunk answers. Null when only one was submitted.")


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
    # Which of the submitted queries this chunk answers; None for a single one.
    query: str | None = None


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
        max_num_results=top_k,
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
        ))
    return hits[:top_k]



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


def _local_search_for_source(query: str, query_vector, source_type: str,
                             top_k: int, exclude_ids: list[str] | None) -> List[_Hit]:
    """One branch of the LOCAL index — the same contract as its vector-store
    twin: the branch's own quota, and the caller's `exclude_ids` dropped before
    the cut rather than after."""
    combined = f"{SOURCETYPE_DOCUMENTATION}-{source_type}"
    rows = local_index.search(
        query_vector, source_type, top_k,
        exclude_ids=set(exclude_ids or ()),
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
    ) for r in rows]
    return hits[:top_k]


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

# `type` argument → TOP_K key
_TYPE_TO_TOP_K = {
    SOURCETYPE_DOCUMENTATION_LANGUAGE: SOURCETYPE_DOC_LANGUAGE,
    SOURCETYPE_DOCUMENTATION_PARADIGM: SOURCETYPE_DOC_PARADIGM,
    SOURCETYPE_DOCUMENTATION_HOWTO: SOURCETYPE_DOC_HOWTO,
}


def retrieve_docs_tool(
    query: str | list[str],
    type: str | None = None,
    exclude_ids: list[str] | None = None,
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
    queries = [query] if isinstance(query, str) else list(query)
    # Identical strings are collapsed BEFORE the budget is divided: otherwise
    # ["x", "x"] halves the share and then hands every chunk to the first copy,
    # leaving the second with nothing and the caller with less than a plain "x"
    # would have returned. Order is preserved.
    queries = list(dict.fromkeys(queries)) if len(queries) > 1 else queries
    requested: tuple[str, ...] | None = None
    n_requested: int | None = None
    try:
        if not queries or any(not isinstance(q, str) or not q.strip() for q in queries):
            raise ValueError("query must be a non-empty string, or a list of them")
        if len(queries) > BATCH_MAX_QUERIES:
            raise ValueError(
                f"a batch carries at most {BATCH_MAX_QUERIES} queries, got {len(queries)}. "
                f"Beyond that each one gets too small a share of the budget to be worth "
                f"asking; split the call.")
        if type is not None and type not in ALLOWED_TYPES:
            raise ValueError(
                f"type must be one of {ALLOWED_TYPES} or null/omitted, got {type!r}"
            )
        requested = (type,) if type else DEFAULT_TYPES
        quotas = TOP_K
        per_call = sum(quotas.get(_TYPE_TO_TOP_K[s], 0) for s in requested)
        # One shared budget, split evenly. A single query is unaffected — its
        # own quota is already under the cap — so batching costs context rather
        # than buying it.
        per_query = per_call if len(queries) == 1 else max(
            1, min(per_call, BATCH_TOTAL_CAP // len(queries)))
        n_requested = per_query * len(queries)

        # The local index is preferred only when it is BOTH switched on and
        # actually loaded, and only if the queries embed. Anything else — no
        # snapshot, a refused snapshot, an embeddings outage — falls through to
        # the vector store, which is how this server has always worked. The
        # fallback is the whole reason the switch is safe to flip.
        vectors = None
        if RETRIEVAL_BACKEND == "local" and local_index.get() is not None:
            try:
                vectors = _embed_queries(queries)
            except Exception as e:  # noqa: BLE001 — degrade to the store, do not fail the call
                log.warning("local backend: could not embed the queries (%s); "
                            "serving this call from the vector store", e)
        backend = "local" if vectors is not None else "store"

        # Every query searched first, then the shared budget handed out. Two
        # passes rather than one because a chunk that answers two of the
        # queries belongs to the one that ranked it higher, and that is not
        # knowable while the first query is still being served.
        found: list[List[_Hit]] = []
        for i, q in enumerate(queries):
            per_query_hits: List[_Hit] = []
            for source_type in requested:
                top_k = quotas.get(_TYPE_TO_TOP_K[source_type], 0)
                if vectors is not None:
                    per_query_hits.extend(_local_search_for_source(
                        q, vectors[i], source_type, top_k, exclude_ids))
                else:
                    per_query_hits.extend(_vs_search_for_source(q, source_type, top_k, exclude_ids))
            # Branches were searched separately, each against its own quota; the
            # caller gets one list, so rank it. Scores are comparable within a
            # backend (all five came from the same one), never across.
            per_query_hits.sort(key=lambda h: -h.score)
            found.append(per_query_hits)

        # Who owns each chunk: the query it scored highest FOR. A chunk is
        # returned ONCE — the caller's context should not hold it twice, and
        # separate calls could never know it was a repeat.
        owner: dict[str, int] = {}
        best: dict[str, float] = {}
        for i, lst in enumerate(found):
            for h in lst:
                if h.id not in best or h.score > best[h.id]:
                    best[h.id], owner[h.id] = h.score, i
        hits: List[_Hit] = []
        for i, (q, lst) in enumerate(zip(queries, found)):
            taken = 0
            for h in lst:
                if taken >= per_query:
                    break
                if owner.get(h.id) != i:
                    continue  # another query answers it better; it goes there
                h.query = q if len(queries) > 1 else None
                hits.append(h)
                taken += 1
        # Handed out per query, returned as one ranked list: `score` is what
        # DocItem promises the order means, and ownership already compared
        # scores across queries, so they are comparable here too.
        hits.sort(key=lambda h: -h.score)
    except Exception as e:  # noqa: BLE001 — log the failed call, then re-raise unchanged
        _log_retrieval(queries, type, n_requested, [], start,
                       ok=False, error=e, exclude_ids=exclude_ids,
                       backend=locals().get("backend", "store"))  # may not be set yet if we failed early
        raise
    # Success path — log OUTSIDE the business `try` so a logging failure can never
    # land in the `except` (false ok=false) or turn a good retrieval into an error.
    # `_log_retrieval` is itself fully guarded and never raises.
    _log_retrieval(queries, type, n_requested, hits, start, ok=True,
                   exclude_ids=exclude_ids, backend=backend)
    return RetrieveDocsOutput(
        docs=[DocItem(id=h.id, source=h.source, text=h.text, score=h.score, query=h.query)
              for h in hits]
    )


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
