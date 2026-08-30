#!/usr/bin/env python3
"""Build the corpus snapshot the server searches locally.

Runs in the same job that indexes the docs (`ragIngestDocs`), from the same
`Section` objects, so the snapshot and the vector store describe one corpus
revision. Embeds every chunk's payload — the SAME bytes the store was given —
and writes one atomic file (see fill/snapshot.py for what is in it and why).

    python3 tools/rag_build_snapshot.py --platform-root <repo> --out <file.npz>

Exit codes: 0 built, 2 setup error (no key, bad root), 3 build failed.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from openai import OpenAI

from fill import snapshot as snap
from fill.chunker import CHUNKER_VERSION, GLOSSARY_VERSION, PREFIX_VERSION, chunk_md
from settings import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL
from tools.rag_ingest_docs import DOCS_SUBDIR, _all_docs, _make_lookups

log = logging.getLogger("rag_build_snapshot")

EXIT_SETUP_ERROR = 2
EXIT_FAILED = 3
# One request per this many chunks. Large enough that 1700 chunks cost ~20
# calls, small enough to stay well inside the per-request token cap.
BATCH = 96


def _corpus_revision(platform_root: Path) -> str:
    """The docs commit this snapshot was built from — the one field that lets a
    stale snapshot be recognized as stale rather than merely different."""
    try:
        out = subprocess.run(
            ["git", "-C", str(platform_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30, check=True)
        return out.stdout.strip()
    except Exception as e:  # noqa: BLE001 — a missing git is not a reason to fail the build
        log.warning("could not read the docs commit (%s); manifest will say 'unknown'", e)
        return "unknown"


def build(platform_root: Path, out: Path, client: OpenAI,
          corpus_revision: str | None = None) -> dict:
    docs_root = platform_root / DOCS_SUBDIR
    if not docs_root.is_dir():
        raise FileNotFoundError(f"docs root not found: {docs_root}")
    source_type_for, slug_for, _source_file_for, _path_for_key = _make_lookups(docs_root)

    rows: list[dict] = []
    for path in _all_docs(docs_root):
        source_type = source_type_for(path)
        slug = slug_for(path)
        for sec in chunk_md(path, source_type, slug):
            rows.append({
                "section_id": sec.section_id,
                "sourceType": sec.source_type,
                "slug": sec.slug,
                "heading_path": sec.heading_path,
                "keywords": sec.keywords,
                "source_url": sec.source_url,
                # The payload, not the raw content: the prefix and keywords are
                # part of what the store embedded, so they must be part of what
                # we embed too, or the two indexes answer differently.
                "text": sec.payload,
            })
    if not rows:
        raise RuntimeError(f"no chunks produced from {docs_root} — refusing to write an empty snapshot")
    log.info("chunked %d sections from %s", len(rows), docs_root)

    vectors = np.empty((len(rows), EMBEDDING_DIMENSIONS), dtype=np.float32)
    for i in range(0, len(rows), BATCH):
        batch = rows[i:i + BATCH]
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL, input=[r["text"] for r in batch])
        # The API returns items in request order, but it also carries an index —
        # trust the index, not the position, so a future change cannot silently
        # pair a vector with the wrong chunk.
        for item in resp.data:
            vectors[i + item.index] = item.embedding
        log.info("embedded %d/%d", min(i + BATCH, len(rows)), len(rows))

    manifest = {
        "embedding_model": EMBEDDING_MODEL,
        # Given by the caller when it knows better than a `git` we may not have
        # (the build runs in a container without one).
        "corpus_revision": corpus_revision or _corpus_revision(platform_root),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chunker_version": CHUNKER_VERSION,
        "glossary_version": GLOSSARY_VERSION,
        "prefix_version": PREFIX_VERSION,
    }
    full = snap.write(out, rows=rows, vectors=vectors, manifest=manifest)
    log.info("wrote %s (%.1f MB): %s", out, out.stat().st_size / 1e6,
             {k: full[k] for k in ("n_chunks", "dimensions", "corpus_revision")})
    return full


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--platform-root", type=Path, required=True,
                   help="checkout of lsfusion/platform (the docs live under docs/)")
    p.add_argument("--out", type=Path, required=True, help="snapshot file to write")
    p.add_argument("--corpus-revision", default=None,
                   help="the docs commit this is built from; read from git when omitted")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        log.error("OPENAI_API_KEY is not set")
        return EXIT_SETUP_ERROR
    try:
        build(args.platform_root, args.out, OpenAI(api_key=key), args.corpus_revision)
    except FileNotFoundError as e:
        log.error("%s", e)
        return EXIT_SETUP_ERROR
    except Exception as e:  # noqa: BLE001 — one exit code for "the artifact was not produced"
        log.exception("snapshot build failed: %s", e)
        return EXIT_FAILED
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
