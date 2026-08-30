"""fill.snapshot — the corpus the server searches, as one file.

WHY A FILE AT ALL. Today the vector store holds the text and its vector
together, so they cannot disagree. Searching locally splits them: the vector we
rank by and the text we hand the assistant are built by one job and read by
another process. Everything here exists to keep those two from drifting —
one artifact, one manifest, atomic publish, and a load that refuses anything it
does not recognize.

WHAT IS IN IT. Per chunk: the payload the embedding was computed FROM (so the
two can be checked against each other), the attributes retrieve_docs returns,
and the unit-normalized vector. Plus a manifest naming the corpus revision, the
embedding model and its dimensions, and the chunker/prefix versions that shaped
the payloads. A snapshot whose manifest does not match what the server expects
is not loaded — a wrong index is worse than no index, because it fails silently.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Bump when the on-disk layout changes in a way an older server cannot read.
SNAPSHOT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class Snapshot:
    """One immutable corpus generation, in memory."""
    manifest: dict
    # Parallel arrays, one entry per chunk, in the SAME order as `vectors`.
    section_ids: list[str]
    source_types: list[str]
    slugs: list[str]
    heading_paths: list[str]
    keywords: list[str]
    source_urls: list[str]
    texts: list[str]
    vectors: np.ndarray  # (n, dim) float32, L2-normalized rows

    def __len__(self) -> int:
        return len(self.section_ids)


def _payload_digest(texts: list[str]) -> str:
    """Fingerprint of what was embedded. Two snapshots with the same digest hold
    the same corpus, whatever else differs."""
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\0")
    return f"sha256:{h.hexdigest()}"


def write(path: Path, *, rows: list[dict], vectors: np.ndarray, manifest: dict) -> dict:
    """Write one snapshot ATOMICALLY: build a temp file beside the target, then
    rename. A reader either sees the previous generation or the new one — never
    half of either, which is the failure that would otherwise be invisible."""
    if len(rows) != len(vectors):
        raise ValueError(f"snapshot: {len(rows)} rows but {len(vectors)} vectors")
    if not rows:
        raise ValueError("snapshot: refusing to write an empty corpus")

    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.all(np.isfinite(vectors)):
        raise ValueError("snapshot: vectors contain NaN or infinity")
    if np.any(norms == 0):
        raise ValueError("snapshot: at least one vector is all zeros")
    vectors = vectors / norms  # normalize ONCE, at build time: search is then a dot product

    texts = [r["text"] for r in rows]
    full = {
        **manifest,
        "format_version": SNAPSHOT_FORMAT_VERSION,
        "n_chunks": len(rows),
        "dimensions": int(vectors.shape[1]),
        "payload_digest": _payload_digest(texts),
        "chunks_per_branch": {
            t: sum(1 for r in rows if r["sourceType"] == t)
            for t in sorted({r["sourceType"] for r in rows})
        },
    }
    # Everything textual goes in ONE json blob, not in numpy string arrays:
    # numpy pads every string to the longest one, which turned what should be a
    # 23 MB snapshot into 99 MB on the real corpus.
    payload = {
        "manifest": full,
        "section_ids": [r["section_id"] for r in rows],
        "source_types": [r["sourceType"] for r in rows],
        "slugs": [r["slug"] for r in rows],
        "heading_paths": [r.get("heading_path") or "" for r in rows],
        "keywords": [r.get("keywords") or "" for r in rows],
        "source_urls": [r.get("source_url") or "" for r in rows],
        "texts": texts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(
            f,
            meta=np.frombuffer(json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                               dtype=np.uint8),
            vectors=vectors,
        )
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return full


def load(path: Path, *, expect_model: str, expect_dimensions: int) -> Snapshot:
    """Read a snapshot and REFUSE anything that does not match what this server
    can search with. A snapshot embedded by another model, or truncated, does
    not raise on its own — it just returns nonsense, so every check here is the
    only thing standing between a bad artifact and silently wrong answers."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(bytes(z["meta"]).decode("utf-8"))
        manifest = meta["manifest"]
        if manifest.get("format_version") != SNAPSHOT_FORMAT_VERSION:
            raise ValueError(
                f"snapshot {path}: format_version {manifest.get('format_version')!r}, "
                f"this server reads {SNAPSHOT_FORMAT_VERSION}")
        if manifest.get("embedding_model") != expect_model:
            raise ValueError(
                f"snapshot {path}: embedded with {manifest.get('embedding_model')!r}, "
                f"server queries with {expect_model!r} — the two are not comparable")
        if manifest.get("dimensions") != expect_dimensions:
            raise ValueError(
                f"snapshot {path}: {manifest.get('dimensions')} dimensions, "
                f"server expects {expect_dimensions}")
        vectors = z["vectors"]
        section_ids = meta["section_ids"]
        texts = meta["texts"]
        if len(section_ids) != manifest.get("n_chunks") or len(vectors) != len(section_ids):
            raise ValueError(f"snapshot {path}: truncated — manifest says "
                             f"{manifest.get('n_chunks')} chunks, file holds {len(section_ids)}")
        if _payload_digest(texts) != manifest.get("payload_digest"):
            raise ValueError(f"snapshot {path}: payload digest mismatch — the texts "
                             f"are not the ones the manifest describes")
        return Snapshot(
            manifest=manifest,
            section_ids=section_ids,
            source_types=meta["source_types"],
            slugs=meta["slugs"],
            heading_paths=meta["heading_paths"],
            keywords=meta["keywords"],
            source_urls=meta["source_urls"],
            texts=texts,
            vectors=np.asarray(vectors, dtype=np.float32),
        )
