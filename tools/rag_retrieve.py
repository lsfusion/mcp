
from __future__ import annotations
from typing import List, Dict, Any

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

# Initialized once per process. In serverless, ensure reuse.
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)


def _chunk_by_words(text: str, max_words: int = 3000) -> List[str]:
    """Split text into ~max_words chunks, space-delimited. No overlap."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def _get_embedding(text: str) -> List[float]:
    """Create embeddings for the query; mean-pool across chunks with L2 normalization per chunk."""
    chunks = _chunk_by_words(text, 3000)
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=chunks)

    if len(resp.data) == 1:
        return list(resp.data[0].embedding)

    dim = len(resp.data[0].embedding)
    acc = [0.0] * dim

    for datum in resp.data:
        vec = list(datum.embedding)
        norm = (sum(v * v for v in vec) ** 0.5) or 1.0
        for i in range(dim):
            acc[i] += vec[i] / norm

    n = len(resp.data)
    return [v / n for v in acc]


def retrieve_docs(query: str) -> List[Dict[str, Any]]:
    """
    Query Pinecone with an OpenAI embedding for the given query.
    Returns a list of { source, text, score } sorted by descending score.
    """
    vec = _get_embedding(query)
    out: List[Dict[str, Any]] = []

    for key, topk in TOP_K.items():
        res = index.query(
            vector=vec,
            top_k=topk,
            include_metadata=True,
            namespace=PINECONE_NAMESPACE or None,
            filter={SOURCETYPE: {"$eq": key}},
        )

        for m in (res.matches or []):
            meta = m.metadata or {}
            out.append({
                "source": key,
                "text": meta.get(TEXT, ""),
                "score": float(m.score or 0.0),
            })

    out.sort(key=lambda d: -d["score"])
    return out
