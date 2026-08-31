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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fill.snapshot import Snapshot, load
from settings import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    SNAPSHOT_MAX_AGE_DAYS,
    SNAPSHOT_PATH,
)

log = logging.getLogger("local_index")

_lock = threading.Lock()
_snapshot: Snapshot | None = None
_tried = False
_stamp: tuple[float, int] | None = None  # (mtime, size) of the file we loaded
_by_branch: dict[str, np.ndarray] = {}  # sourceType -> row indices into the snapshot


def _stamp_of(path: Path) -> tuple[float, int] | None:
    try:
        st = path.stat()
        return (st.st_mtime, st.st_size)
    except OSError:
        return None


def get() -> Snapshot | None:
    """The loaded snapshot, or None if there isn't a usable one. Safe to call
    from several threads.

    A new snapshot is picked up WITHOUT a restart: the ingest job publishes one
    on every docs change (2-3 days a week), and restarting the server for each
    would drop every connected MCP session. So each call stats the file — one
    syscall — and reloads when it has been replaced. A reload that fails keeps
    the snapshot already in memory: a corpus one generation old beats none.
    """
    global _snapshot, _tried, _stamp
    path = Path(SNAPSHOT_PATH)
    stamp = _stamp_of(path) if SNAPSHOT_PATH else None
    if (_snapshot is not None or _tried) and stamp == _stamp:
        return _snapshot
    with _lock:
        stamp = _stamp_of(path) if SNAPSHOT_PATH else None
        if (_snapshot is not None or _tried) and stamp == _stamp:
            return _snapshot
        _tried = True
        if stamp is None:
            if _snapshot is None:
                log.info("no snapshot at %s — retrieval will use the vector store",
                         SNAPSHOT_PATH)
            else:
                log.warning("snapshot %s has disappeared — keeping the one already "
                            "loaded", SNAPSHOT_PATH)
            return _snapshot
        try:
            snap = load(path, expect_model=EMBEDDING_MODEL,
                        expect_dimensions=EMBEDDING_DIMENSIONS)
        except Exception as e:  # noqa: BLE001 — a bad artifact must not take the server down
            log.error("snapshot %s refused (%s) — %s", path, e,
                      "keeping the one already loaded" if _snapshot is not None
                      else "retrieval falls back to the vector store")
            # Remember the stamp anyway, or every single call re-reads 23 MB
            # just to fail again on the same bad file.
            _stamp = stamp
            return _snapshot
        _snapshot = snap
        _stamp = stamp
        _by_branch.clear()
        for t in sorted(set(snap.source_types)):
            _by_branch[t] = np.array(
                [i for i, s in enumerate(snap.source_types) if s == t], dtype=np.int64)
        log.info("snapshot loaded: %d chunks, %s, revision %s, built %s, per branch %s",
                 len(snap), snap.manifest.get("embedding_model"),
                 snap.manifest.get("corpus_revision"), snap.manifest.get("built_at"),
                 snap.manifest.get("chunks_per_branch"))
        age = _age_days(snap)
        if age is not None and age > SNAPSHOT_MAX_AGE_DAYS:
            log.warning(
                "snapshot is %.1f days old (built %s, revision %s) — the docs have "
                "almost certainly moved since; the delivery job is not running",
                age, snap.manifest.get("built_at"), snap.manifest.get("corpus_revision"))
        elif age is None:
            log.warning("snapshot carries no build time — it was built by an older "
                        "builder, and its age cannot be checked")
        return _snapshot


def _age_days(snap: Snapshot) -> float | None:
    """How long ago this snapshot was built, or None if it does not say."""
    built = snap.manifest.get("built_at")
    if not built:
        return None
    try:
        when = datetime.fromisoformat(built)
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - when).total_seconds() / 86400.0


def reset_for_tests() -> None:
    """Drop the cached snapshot. Tests only — production loads once and keeps it."""
    global _snapshot, _tried, _stamp
    with _lock:
        _snapshot, _tried, _stamp = None, False, None
        _by_branch.clear()


def search(query_vector: np.ndarray, source_type: str, top_k: int,
           exclude_ids: set[str] | None = None,
) -> list[dict]:
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
