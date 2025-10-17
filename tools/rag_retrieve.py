from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field
import os
import json

from openai import OpenAI
from pinecone import Pinecone

from settings import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX,
    PINECONE_NAMESPACE,
    EMBEDDING_MODEL,
    TEXT,
    SOURCETYPE,
    TOP_K,
)

# ====== Pydantic models ======
class DocItem(BaseModel):
    """Single retrieved chunk from the RAG knowledge base."""
    source: str = Field(..., description="Chunk origin (e.g. docs, howto, tutorial).")
    text: str = Field(..., description="Retrieved text snippet.")
    score: float = Field(..., description="Similarity score (higher = more relevant).")


class RetrieveDocsOutput(BaseModel):
    """List of retrieved chunks sorted by relevance."""
    docs: List[DocItem] = Field(
        default_factory=list,
        description="Relevant chunks returned from the RAG store."
    )


# ====== Initialization ======
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# --- DIAGNOSTICS (one-time) ---
def _mask(s: str | None, keep: int = 4) -> str:
    if not s:
        return "<empty>"
    return ("*" * max(0, len(s) - keep)) + s[-keep:]

def _print_index_stats():
    try:
        stats = index.describe_index_stats()
        print("[diagnostics] describe_index_stats():")
        # pretty but safe
        print(json.dumps(stats, indent=2)[:4000])  # truncate to avoid huge dumps
        if PINECONE_NAMESPACE:
            ns = stats.get("namespaces", {}).get(PINECONE_NAMESPACE, {})
            print(f"[diagnostics] vectors in namespace '{PINECONE_NAMESPACE}': {ns.get('vector_count')}")
        else:
            total = stats.get("total_vector_count")
            print(f"[diagnostics] total_vector_count: {total}")
    except Exception as e:
        print(f"[diagnostics] describe_index_stats() error: {e!r}")

print("[diagnostics] Runtime config:")
print(f"  PINECONE_INDEX      = {PINECONE_INDEX!r}")
print(f"  PINECONE_NAMESPACE  = {PINECONE_NAMESPACE!r}")
print(f"  EMBEDDING_MODEL     = {EMBEDDING_MODEL!r}")
print(f"  SOURCETYPE key      = {SOURCETYPE!r}")
print(f"  TOP_K               = {TOP_K!r}")
print(f"  OPENAI_API_KEY set? = {bool(OPENAI_API_KEY)} ({_mask(os.getenv('OPENAI_API_KEY'))})")
print(f"  PINECONE_API_KEY?   = {bool(PINECONE_API_KEY)} ({_mask(os.getenv('PINECONE_API_KEY'))})")

_print_index_stats()


def _chunk_by_words(text: str, max_words: int = 3000) -> List[str]:
    """
    Split text into space-delimited chunks with up to `max_words` words.
    Used to keep embedding requests within model limits.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def _get_embedding(text: str) -> List[float]:
    """
    Generate an averaged embedding vector for the given text.
    Each chunk is L2-normalized before averaging.
    """
    chunks = _chunk_by_words(text, 3000)
    try:
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=chunks)
    except Exception as e:
        print(f"[embedding] ERROR calling OpenAI embeddings: {e!r}")
        raise

    # Single chunk: return as-is
    if len(resp.data) == 1:
        vec = list(resp.data[0].embedding)
        print(f"[embedding] len={len(vec)} first_values={vec[:8]}")
        return vec

    # Average across normalized chunk embeddings
    dim = len(resp.data[0].embedding)
    acc = [0.0] * dim
    for datum in resp.data:
        vec = list(datum.embedding)
        norm = (sum(v * v for v in vec) ** 0.5) or 1.0
        for i in range(dim):
            acc[i] += vec[i] / norm

    n = len(resp.data)
    avg = [v / n for v in acc]
    print(f"[embedding] len={len(avg)} first_values={avg[:8]} (averaged from {n} chunks)")
    return avg


def retrieve_docs(query: str) -> RetrieveDocsOutput:
    """
    Fetch prioritized chunks from your RAG store—documentation, how-tos, tutorials and articles—
    based on a single search query.
    """
    print(f"[retrieve_docs] Query: {query!r}")

    vec = _get_embedding(query)
    items: List[DocItem] = []

    # Query each source type defined in TOP_K
    for source_type, top_k in TOP_K.items():
        pinecone_filter = {SOURCETYPE: {"$eq": source_type}}
        try:
            res = index.query(
                vector=vec,
                top_k=top_k,
                include_metadata=True,
                namespace=PINECONE_NAMESPACE or None,
                filter=pinecone_filter,
            )
        except Exception as e:
            print(f"[retrieve_docs] ERROR querying Pinecone for source_type={source_type!r}: {e!r}")
            continue

        matches = res.matches or []
        # Per-source diagnostics
        ex_id = getattr(matches[0], "id", None) if matches else None
        ex_meta = getattr(matches[0], "metadata", None) if matches else None
        print(
            f"[retrieve_docs] source_type={source_type!r} "
            f"top_k={top_k} filter={pinecone_filter} "
            f"matches={len(matches)} example_id={ex_id!r} "
            f"example_meta_keys={list(ex_meta.keys()) if isinstance(ex_meta, dict) else None}"
        )

        for match in matches:
            meta = match.metadata or {}
            items.append(DocItem(
                source=source_type,
                text=meta.get(TEXT, "") or "",
                score=float(match.score or 0.0),
            ))

    # Sort results by descending score
    items.sort(key=lambda d: -d.score)
    output = RetrieveDocsOutput(docs=items)

    print(f"[retrieve_docs] Retrieved {len(items)} document(s) total.")
    # dump small preview to spot empty texts or wrong sources
    for i, d in enumerate(items[:10], 1):
        preview = (d.text or "")[:120].replace("\n", " ")
        print(f"  {i}. [{d.source}] score={d.score:.4f} — {preview!r}")

    # final cross-check: if zero docs, show index stats again (maybe namespace mismatch)
    if not items:
        print("[retrieve_docs] No documents found — re-checking index stats and namespace...")
        _print_index_stats()
        print(f"[retrieve_docs] Effective namespace used: {PINECONE_NAMESPACE or None!r}")

    return output
