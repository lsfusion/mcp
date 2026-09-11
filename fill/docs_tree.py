"""fill.docs_tree — where a documentation file lives, and what that means.

The layout is `docs/<type>/en/<stem>.md`, and the type FOLDER is the single
source of truth: there is no manifest, and a path that does not fit the shape
raises rather than being guessed at.

This lives apart from any one caller because the snapshot builder and the
ingest driver both need it, and the second is on its way out. Keeping the
layout rules here means the builder does not import from a module that is
being deleted around it — which is the way this deletion would otherwise go
wrong: request-time tests stay green against a snapshot that already exists,
and the next rebuild is the thing that fails.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from fill.chunker import INDEXED_TYPES, SOURCE_TYPES, SourceType

# The documentation tree, relative to the platform checkout.
DOCS_SUBDIR = Path("docs")


def all_docs(docs_root: Path) -> list[Path]:
    # Type-first: ingest the English slice docs/<type>/en/*.md only. `docs/images`
    # and `docs/<type>/AGENTS.md` are skipped (neither sits under <type>/en).
    out: list[Path] = []
    for t in sorted(INDEXED_TYPES):
        en = docs_root / t / "en"
        if en.is_dir():
            out += [p for p in en.rglob("*.md") if p.is_file()]
    return sorted(out)


def make_lookups(docs_root: Path) -> tuple[
    Callable[[Path], SourceType],
    Callable[[Path], str],
    Callable[[Path], str],
    Callable[[str], Path],
]:
    """Resolve, for a doc at docs/<type>/en/<stem>.md:
      - source_type_for → the type folder (`language`/`paradigm`/…)
      - slug_for        → the filename stem
      - source_file_for → the logical key "<type>/<stem>.md" (language segment
        dropped, so the key is stable across the en/ru<->type/lang move and the
        vector store sees no change)
      - path_for_key    → inverse of source_file_for (used to re-find stale files)

    The type folder is the single source of truth (no manifest). A path that is
    not docs/<type>/en/<file>.md raises — surfacing as a per-file setup error in
    `fill.ingest`, never aborting the whole run.
    """
    def slug_for(path: Path) -> str:
        return path.stem

    def source_type_for(path: Path) -> SourceType:
        try:
            rel = path.relative_to(docs_root)
        except ValueError:
            raise ValueError(f"{path} is not under docs root {docs_root}")
        parts = rel.parts
        if len(parts) < 3 or parts[1] != "en":
            raise ValueError(
                f"{rel}: expected <type>/en/<file>.md "
                f"(type in {sorted(SOURCE_TYPES)})"
            )
        folder = parts[0]
        if folder not in SOURCE_TYPES:
            raise ValueError(
                f"{rel}: folder {folder!r} is not a valid sourceType "
                f"{sorted(SOURCE_TYPES)}"
            )
        return folder  # type: ignore[return-value]

    def source_file_for(path: Path) -> str:
        rel = path.relative_to(docs_root)  # <type>/en/<stem>.md
        return f"{rel.parts[0]}/{path.name}"

    def path_for_key(key: str) -> Path:
        t, _, name = key.partition("/")  # "<type>/<stem>.md"
        return docs_root / t / "en" / name

    return source_type_for, slug_for, source_file_for, path_for_key
