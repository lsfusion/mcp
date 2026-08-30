"""tools.local_index — searching the corpus in this process.

The whole search is a matrix multiply: every chunk is a unit vector, the query
is a unit vector, and the ranking is their dot products. 1704 chunks at 3072
dimensions is 20 MB and a few milliseconds, so there is no approximation and no
index structure — every chunk is compared, every time. What that buys over an
approximate index is that the answer cannot depend on how the neighbourhood
graph happened to be built.

The snapshot is loaded once, lazily, and held. A failure to load is not fatal:
`get()` returns None and the caller falls back to the vector store, which is
what this server did before the snapshot existed. Load failures are logged
loudly BUT ONLY ONCE — a broken artifact should be visible in the log, not
repeated on every request.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from fill.snapshot import Snapshot, load
from settings import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, SNAPSHOT_PATH

log = logging.getLogger("local_index")

_lock = threading.Lock()
_snapshot: Snapshot | None = None
_tried = False
_by_branch: dict[str, np.ndarray] = {}  # sourceType -> row indices into the snapshot


def get() -> Snapshot | None:
    """The loaded snapshot, or None if there isn't a usable one. Cheap after
    the first call; safe to call from several threads."""
    global _snapshot, _tried
    if _snapshot is not None or _tried:
        return _snapshot
    with _lock:
        if _snapshot is not None or _tried:
            return _snapshot
        _tried = True
        path = Path(SNAPSHOT_PATH)
        if not SNAPSHOT_PATH or not path.is_file():
            log.info("no snapshot at %s — retrieval will use the vector store", SNAPSHOT_PATH)
            return None
        try:
            snap = load(path, expect_model=EMBEDDING_MODEL,
                        expect_dimensions=EMBEDDING_DIMENSIONS)
        except Exception as e:  # noqa: BLE001 — a bad artifact must not take the server down
            log.error("snapshot %s refused (%s) — retrieval falls back to the vector store",
                      path, e)
            return None
        _snapshot = snap
        for t in sorted(set(snap.source_types)):
            _by_branch[t] = np.array(
                [i for i, s in enumerate(snap.source_types) if s == t], dtype=np.int64)
        log.info("snapshot loaded: %d chunks, %s, revision %s, per branch %s",
                 len(snap), snap.manifest.get("embedding_model"),
                 snap.manifest.get("corpus_revision"), snap.manifest.get("chunks_per_branch"))
        return _snapshot


def reset_for_tests() -> None:
    """Drop the cached snapshot. Tests only — production loads once and keeps it."""
    global _snapshot, _tried
    with _lock:
        _snapshot, _tried = None, False
        _by_branch.clear()


def search(query_vector: np.ndarray, source_type: str, top_k: int,
           exclude_ids: set[str] | None = None,
           exclude_slugs: set[str] | None = None) -> list[dict]:
    """Top `top_k` chunks of ONE branch, best first.

    Exclusions are applied BEFORE the cut, not after — dropping them from a
    finished top-k would silently return fewer chunks than asked for, which is
    exactly what `exclude_ids` exists to avoid when the caller pages deeper.
    """
    snap = get()
    if snap is None or top_k <= 0:
        return []
    rows = _by_branch.get(source_type)
    if rows is None or len(rows) == 0:
        return []
    scores = snap.vectors[rows] @ query_vector  # both sides are unit vectors => cosine
    order = np.argsort(-scores)
    out: list[dict] = []
    for pos in order:
        i = int(rows[pos])
        sid = snap.section_ids[i]
        if exclude_ids and sid in exclude_ids:
            continue
        if exclude_slugs and snap.slugs[i] in exclude_slugs:
            continue
        out.append({
            "section_id": sid,
            "sourceType": snap.source_types[i],
            "slug": snap.slugs[i],
            "heading_path": snap.heading_paths[i],
            "keywords": snap.keywords[i],
            "text": snap.texts[i],
            "score": float(scores[pos]),
        })
        if len(out) >= top_k:
            break
    return out
