from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field

from openai import OpenAI

from settings import (
    OPENAI_API_KEY,
    RAG_VECTOR_STORE_ID,
    SOURCETYPE_DOCUMENTATION,
    SOURCETYPE_DOCUMENTATION_PARADIGM,
    SOURCETYPE_DOCUMENTATION_LANGUAGE,
    SOURCETYPE_DOC_PARADIGM,
    SOURCETYPE_DOC_LANGUAGE,
    SOURCETYPE,
    TOP_K,
)


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


def _vs_search_for_source(query: str, source_type: str, top_k: int) -> List[DocItem]:
    """Search the OpenAI Vector Store for chunks tagged with `sourceType=source_type`.

    `source_type` is the bare manifest value ("language" / "paradigm"). The
    returned `DocItem.source` uses the combined "documentation-<type>" form
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
    items: List[DocItem] = []
    for hit in resp.data:
        text_parts = [c.text for c in (hit.content or []) if getattr(c, "type", None) == "text"]
        items.append(DocItem(
            source=combined,
            text="\n".join(text_parts),
            score=float(hit.score or 0.0),
        ))
    return items


ALLOWED_TYPES = (
    SOURCETYPE_DOCUMENTATION_LANGUAGE,
    SOURCETYPE_DOCUMENTATION_PARADIGM,
)

# `type` argument → (bare sourceType filter, TOP_K key)
_TYPE_TO_TOP_K = {
    SOURCETYPE_DOCUMENTATION_LANGUAGE: SOURCETYPE_DOC_LANGUAGE,
    SOURCETYPE_DOCUMENTATION_PARADIGM: SOURCETYPE_DOC_PARADIGM,
}


def retrieve_docs_tool(query: str, type: str | None = None) -> RetrieveDocsOutput:
    """Retrieve chunks from the OpenAI Vector Store populated by the
    `ragIngestDocs` Jenkins pipeline.

    `type` filters by chunk sourceType:
      * omitted / null — search both `language` and `paradigm`, merge results.
      * `"language"` — only language reference chunks.
      * `"paradigm"` — only paradigm / conceptual chunks.

    The store only holds English (`docs/en/`) content. Cross-lingual
    embeddings make non-English queries work, but English wording is
    preferred for best recall.
    """
    if type is not None and type not in ALLOWED_TYPES:
        raise ValueError(
            f"type must be one of {ALLOWED_TYPES} or null/omitted, got {type!r}"
        )
    requested = (type,) if type else ALLOWED_TYPES
    items: List[DocItem] = []
    for source_type in requested:
        items.extend(_vs_search_for_source(
            query, source_type, TOP_K.get(_TYPE_TO_TOP_K[source_type], 0)))
    items.sort(key=lambda d: -d.score)
    return RetrieveDocsOutput(docs=items)
