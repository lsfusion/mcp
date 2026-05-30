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
    TOP_K,
    QUERY_LOG_MAX_CHARS,
    ERROR_LOG_MAX_CHARS,
)
from tools.event_log import emit


class DocItem(BaseModel):
    """Single retrieved chunk from the RAG knowledge base."""
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

    `source` is the combined "documentation-<type>" branch label returned to
    consumers; `file_id`/`filename` are the Vector Store identifiers logged so
    a later triage can compare the served result against the live store
    (MCP-FEEDBACK-PLAN.md, Phase A). They are NOT exposed in the tool response.
    """
    source: str
    text: str
    score: float
    file_id: str | None
    filename: str | None


def _vs_search_for_source(query: str, source_type: str, top_k: int) -> List[_Hit]:
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
        filters={"type": "eq", "key": SOURCETYPE, "value": source_type},
        rewrite_query=False,
    )
    combined = f"{SOURCETYPE_DOCUMENTATION}-{source_type}"
    hits: List[_Hit] = []
    for hit in resp.data:
        text_parts = [c.text for c in (hit.content or []) if getattr(c, "type", None) == "text"]
        hits.append(_Hit(
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

# `type` argument → (bare sourceType filter, TOP_K key)
_TYPE_TO_TOP_K = {
    SOURCETYPE_DOCUMENTATION_LANGUAGE: SOURCETYPE_DOC_LANGUAGE,
    SOURCETYPE_DOCUMENTATION_PARADIGM: SOURCETYPE_DOC_PARADIGM,
    SOURCETYPE_DOCUMENTATION_HOWTO: SOURCETYPE_DOC_HOWTO,
    SOURCETYPE_DOCUMENTATION_BRIEF: SOURCETYPE_DOC_BRIEF,
    SOURCETYPE_DOCUMENTATION_RULES: SOURCETYPE_DOC_RULES,
}


def retrieve_docs_tool(query: str, type: str | None = None) -> RetrieveDocsOutput:
    """Retrieve chunks from the OpenAI Vector Store populated by the
    `ragIngestDocs` Jenkins pipeline.

    `type` filters by chunk sourceType (the docs folder):
      * omitted / null — search all branches (language, paradigm, how-to, brief,
        rules) with a per-branch quota and merge results by score.
      * one of `language` / `paradigm` / `how-to` / `brief` / `rules` — only that
        branch.

    The store only holds English (`docs/en/`) content. Cross-lingual
    embeddings make non-English queries work, but English wording is
    preferred for best recall.
    """
    start = time.monotonic()
    requested: tuple[str, ...] | None = None
    try:
        if type is not None and type not in ALLOWED_TYPES:
            raise ValueError(
                f"type must be one of {ALLOWED_TYPES} or null/omitted, got {type!r}"
            )
        requested = (type,) if type else ALLOWED_TYPES
        hits: List[_Hit] = []
        for source_type in requested:
            hits.extend(_vs_search_for_source(
                query, source_type, TOP_K.get(_TYPE_TO_TOP_K[source_type], 0)))
        hits.sort(key=lambda h: -h.score)
    except Exception as e:  # noqa: BLE001 — log the failed call, then re-raise unchanged
        _log_retrieval(query, type, requested, [], start, ok=False, error=e)
        raise
    # Success path — log OUTSIDE the business `try` so a logging failure can never
    # land in the `except` (false ok=false) or turn a good retrieval into an error.
    # `_log_retrieval` is itself fully guarded and never raises.
    _log_retrieval(query, type, requested, hits, start, ok=True)
    return RetrieveDocsOutput(
        docs=[DocItem(source=h.source, text=h.text, score=h.score) for h in hits]
    )


def _log_retrieval(
    query: str,
    type_arg: str | None,
    requested: tuple[str, ...] | None,
    hits: List[_Hit],
    start: float,
    *,
    ok: bool,
    error: BaseException | None = None,
) -> None:
    """Build and emit one `retrieve_docs` event (no chunk text). NEVER raises —
    logging must not break `retrieve_docs` even if field-building itself fails."""
    try:
        fields: dict = {
            "query": (query or "")[:QUERY_LOG_MAX_CHARS],
            "type": type_arg,
            # Total chunk budget requested across the searched branches (sum of
            # per-branch TOP_K), NOT the branch count — e.g. 15 when type is omitted.
            "n_requested": (
                sum(TOP_K.get(_TYPE_TO_TOP_K[s], 0) for s in requested) if requested else None
            ),
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
