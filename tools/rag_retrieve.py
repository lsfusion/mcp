from __future__ import annotations
import time
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
    TOP_K,
    TYPED_TOP_K,
    QUERY_LOG_MAX_CHARS,
    ERROR_LOG_MAX_CHARS,
)
from tools.event_log import emit


class DocItem(BaseModel):
    """Single retrieved chunk from the RAG knowledge base."""
    id: str = Field(..., description="Stable chunk id; pass it in `exclude_ids` to keep this chunk out of a follow-up retrieve_docs call.")
    source: str = Field(..., description="Chunk origin (e.g. documentation-language, documentation-paradigm).")
    text: str = Field(..., description="Retrieved text snippet.")
    score: float = Field(..., description="Similarity score (higher = more relevant).")


class RetrieveDocsOutput(BaseModel):
    """List of retrieved chunks sorted by relevance."""
    docs: List[DocItem] = Field(
        default_factory=list,
        description="Relevant chunks returned from the RAG store."
    )


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
            # (fill/ingest.py:_section_attributes), so a hit without one means
            # the index holds foreign/stale files. Fail loudly instead of
            # serving a chunk that cannot be identified or excluded later.
            raise ValueError(
                f"vector store chunk without a '{SECTION_ID}' attribute "
                f"(file_id={getattr(hit, 'file_id', None)!r}, "
                f"filename={getattr(hit, 'filename', None)!r}) — corrupted index, re-run ragIngestDocs"
            )
        hits.append(_Hit(
            id=str(section_id),
            source=combined,
            text="\n".join(text_parts),
            score=float(hit.score or 0.0),
            file_id=getattr(hit, "file_id", None),
            filename=getattr(hit, "filename", None),
        ))
    return hits


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
        for source_type in requested:
            hits.extend(_vs_search_for_source(
                query, source_type, quotas.get(_TYPE_TO_TOP_K[source_type], 0),
                exclude_ids))
        hits.sort(key=lambda h: -h.score)
    except Exception as e:  # noqa: BLE001 — log the failed call, then re-raise unchanged
        _log_retrieval(query, type, n_requested, [], start,
                       ok=False, error=e, exclude_ids=exclude_ids)
        raise
    # Success path — log OUTSIDE the business `try` so a logging failure can never
    # land in the `except` (false ok=false) or turn a good retrieval into an error.
    # `_log_retrieval` is itself fully guarded and never raises.
    _log_retrieval(query, type, n_requested, hits, start, ok=True, exclude_ids=exclude_ids)
    return RetrieveDocsOutput(
        docs=[DocItem(id=h.id, source=h.source, text=h.text, score=h.score) for h in hits]
    )


def _log_retrieval(
    query: str,
    type_arg: str | None,
    n_requested: int | None,
    hits: List[_Hit],
    start: float,
    *,
    ok: bool,
    error: BaseException | None = None,
    exclude_ids: list[str] | None = None,
) -> None:
    """Build and emit one `retrieve_docs` event (no chunk text). NEVER raises —
    logging must not break `retrieve_docs` even if field-building itself fails."""
    try:
        fields: dict = {
            "query": (query or "")[:QUERY_LOG_MAX_CHARS],
            "type": type_arg,
            # Total chunk budget actually requested across the searched branches
            # (computed by the caller from the quota table that call used), NOT
            # the branch count — e.g. 15 when type is omitted.
            "n_requested": n_requested,
            # Count only: the ids are caller-supplied chunk ids, and the log
            # keeps no chunk identity the caller did not get from us anyway.
            "n_excluded": len(exclude_ids or []),
            "n_results": len(hits),
            "top_score": (hits[0].score if hits else None),
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
            fields["error_message"] = str(error)[:ERROR_LOG_MAX_CHARS]
        emit("retrieve_docs", fields, stream="retrieval", ok=ok)
    except Exception:  # noqa: BLE001 — best-effort; never propagate into retrieve_docs
        pass
