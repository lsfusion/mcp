from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field

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
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=chunks)

    # Single chunk: return as-is
    if len(resp.data) == 1:
        return list(resp.data[0].embedding)

    # Average across normalized chunk embeddings
    dim = len(resp.data[0].embedding)
    acc = [0.0] * dim
    for datum in resp.data:
        vec = list(datum.embedding)
        norm = (sum(v * v for v in vec) ** 0.5) or 1.0
        for i in range(dim):
            acc[i] += vec[i] / norm

    n = len(resp.data)
    return [v / n for v in acc]


def retrieve_docs(query: str) -> RetrieveDocsOutput:
    """
    Fetch prioritized chunks from your RAG store—documentation, how-tos, tutorials and articles—
    based on a single search query.
    """
    vec = _get_embedding(query)
    items: List[DocItem] = []

    # Query each source type defined in TOP_K
    for source_type, top_k in TOP_K.items():
        res = index.query(
            vector=vec,
            top_k=top_k,
            include_metadata=True,
            namespace=PINECONE_NAMESPACE or None,
            filter={SOURCETYPE: {"$eq": source_type}},
        )

        for match in (res.matches or []):
            meta = match.metadata or {}
            items.append(DocItem(
                source=source_type,
                text=meta.get(TEXT, "") or "",
                score=float(match.score or 0.0),
            ))

    # Sort results by descending score
    items.sort(key=lambda d: -d.score)
    output = RetrieveDocsOutput(docs=items)

    return output
