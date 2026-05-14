"""Markdown → list[Section] for OpenAI Vector Store upload.

Implements RAG-PLAN.md §"Chunking algorithm":

  - LangChain MarkdownHeaderTextSplitter on H1/H2/H3 (strip_headers=True)
  - Frontmatter is stripped via python-frontmatter; YAML `title` becomes the
    synthetic H1 so the preamble paragraph(s) get a heading_path.
  - For sourceType="how-to", adjacent `### Task` + `### Solution` pairs
    (en or `### Задача`/`### Решение` ru) under the same H2 are merged into
    one Section with section_id `{slug}::task-NNN`.
  - Oversized parts are passed through a token-aware recursive splitter and
    become `{section_id}::part-NN` sub-sections (overlap preserved).
  - Deterministic prefix `# {sourceType}: {heading_path}\n\n{content}` is
    prepended; full payload feeds OpenAI's chunking_strategy with
    `max_chunk_size_tokens=4096`.
  - section_payload_hash = sha256(JSON{content, sourceType, heading_path,
    chunker_version, glossary_version, prefix_version, source_url_version}).

One-section-one-chunk invariant: with MAX_SECTION_TOKENS=2000 locally and
OpenAI cap 4096, the secondary splitter guarantees each Section's payload
fits inside one OpenAI chunk. Test: test_invariant_no_payload_exceeds_safe_limit.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import frontmatter
import tiktoken
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from fill.config import build_source_url
from fill.versions import (
    CHUNKER_VERSION,
    GLOSSARY_VERSION,
    PREFIX_VERSION,
    SOURCE_URL_VERSION,
)

SourceType = Literal["paradigm", "language", "how-to"]

# ───────────────────────────── tuning constants ──────────────────────────────

# Local cap measured with tiktoken's cl100k_base. The actual embedder may use
# a different tokenizer (e.g., o200k_base) — the OpenAI-side limit (4096) gives
# a ~2x safety buffer to absorb tokenizer disagreement.
#
# MAX_SECTION_TOKENS is a soft TARGET: secondary-split aims for sub-payloads
# in this range, but with pathologically long heading paths (>1900 tokens of
# prefix) the actual payload can land between MAX_SECTION_TOKENS and
# OPENAI_CHUNK_LIMIT. The hard contract is OPENAI_CHUNK_LIMIT — payloads
# strictly above it raise RuntimeError. For realistic lsFusion docs all
# heading paths are < 100 tokens, well inside the target.
MAX_SECTION_TOKENS: int = 2000
OPENAI_CHUNK_LIMIT: int = 4096

# Budget reserved for the deterministic prefix `# {sourceType}: {heading_path}`
# inside each section. Real prefixes are typically 15–40 tokens; 100 is generous.
PREFIX_TOKEN_BUDGET: int = 100

SECONDARY_OVERLAP_TOKENS: int = 100

# Tokenizer for size measurement. Plan acknowledges this is not necessarily
# the embedding model's tokenizer — see plan §"One-section-one-chunk invariant".
TOKENIZER = tiktoken.get_encoding("cl100k_base")

# Header-aware splitter — main entry point.
_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
    strip_headers=True,
)

# Token-aware recursive secondary splitter for oversized sections.
# `chunk_size` is in tokens because we pass `length_function=count_tokens`.
_SECONDARY_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=MAX_SECTION_TOKENS - PREFIX_TOKEN_BUDGET,
    chunk_overlap=SECONDARY_OVERLAP_TOKENS,
    length_function=lambda s: len(TOKENIZER.encode(s)),
    # Try semantic boundaries first; fall back to whitespace, then any char.
    separators=["\n\n", "\n", ". ", " ", ""],
)

# How-to Task/Solution label sets (lowercase, en+ru).
_TASK_LABELS = {"task", "задача"}
_SOLUTION_LABELS = {"solution", "решение"}

# Slug characters allowed in section_id paths (after kebab conversion).
_KEBAB_ALLOWED = re.compile(r"[^a-z0-9_\-=.]")


# ───────────────────────────── public types ──────────────────────────────────


@dataclass
class Section:
    """One indexable unit. After secondary-split, distinct objects may share
    a parent section_id but have differing `::part-NN` suffixes.

    `raw_content` excludes the prefix; full upload payload is `payload`.
    `section_payload_hash` is the primary reindex signal (see plan).
    """

    slug: str
    section_id: str
    section_name: str           # human-readable last segment of heading_path
    heading_path: str           # e.g. "AGGR operator > Syntax"
    source_url: str
    source_type: SourceType
    raw_content: str

    @property
    def prefix(self) -> str:
        return f"# {self.source_type}: {self.heading_path}\n\n"

    @property
    def payload(self) -> str:
        return self.prefix + self.raw_content

    @property
    def section_payload_hash(self) -> str:
        # section_id is included so dup-N / part-NN siblings with identical
        # raw_content produce distinct hashes. Otherwise two `::part-NN`
        # sub-sections whose secondary-split overlap windows happen to
        # produce equal strings would collide on the hash, breaking the
        # (section_id, hash) → 1:1 identity assumption used downstream.
        h = hashlib.sha256(
            json.dumps(
                {
                    "section_id": self.section_id,
                    "content": self.raw_content,
                    "sourceType": self.source_type,
                    "heading_path": self.heading_path,
                    "chunker_version": CHUNKER_VERSION,
                    "glossary_version": GLOSSARY_VERSION,
                    "prefix_version": PREFIX_VERSION,
                    "source_url_version": SOURCE_URL_VERSION,
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        )
        return f"sha256:{h.hexdigest()}"


# ───────────────────────────── helpers ───────────────────────────────────────


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


def kebab_case(name: str) -> str:
    """Lowercase + spaces→`-` + strip non-allowed chars. Mirrors a simple
    subset of Docusaurus' header-auto-anchor heuristic. Used for section_id
    construction only; not for URLs."""
    s = name.strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = _KEBAB_ALLOWED.sub("", s)
    return s or "section"


def _heading_segments(metadata: dict, default_h1: str) -> list[str]:
    """Build the breadcrumb list from a Document's metadata dict."""
    segs = []
    for k in ("h1", "h2", "h3"):
        v = metadata.get(k)
        if v:
            segs.append(v)
    if not segs:
        # Should not happen after synthetic H1, but defensive fallback.
        segs = [default_h1]
    return segs


def _deepest_header(metadata: dict) -> str | None:
    for k in ("h3", "h2", "h1"):
        v = metadata.get(k)
        if v:
            return v
    return None


def _kebab_chain(metadata: dict) -> str:
    """Compact section_id suffix derived from deepest H3 (or H2/H1 fallback)."""
    h = _deepest_header(metadata)
    return kebab_case(h) if h else "_root"


# ───────────────────────────── how-to grouping ───────────────────────────────


def _is_task(part) -> bool:
    h3 = part.metadata.get("h3", "")
    return h3.strip().lower() in _TASK_LABELS


def _is_solution(part) -> bool:
    h3 = part.metadata.get("h3", "")
    return h3.strip().lower() in _SOLUTION_LABELS


def _group_task_solution_pairs(parts: list) -> list:
    """Merge adjacent Task → Solution Documents into one. Same H2 (or both
    missing H2) required for adjacency. Edges:
      - Task without following Solution → standalone
      - Solution without preceding Task → standalone
      - Different H2 between → both standalone
    """
    out: list = []
    i = 0
    while i < len(parts):
        a = parts[i]
        b = parts[i + 1] if i + 1 < len(parts) else None
        if (
            _is_task(a)
            and b is not None
            and _is_solution(b)
            and a.metadata.get("h2") == b.metadata.get("h2")
        ):
            merged_content = f"{a.page_content}\n\n{b.page_content}"
            # Synthesize a Document-like wrapper with merged content and a
            # marker we read later to assign `task-NNN` section_ids.
            # Drop the h3 ("Task") from metadata — the merged section's
            # heading_path should stop at the H2 (e.g. "Title > Example 1"),
            # not point to either half of the pair.
            merged_meta = dict(a.metadata)
            merged_meta.pop("h3", None)
            merged_meta["_task_pair"] = True
            out.append(_DocLike(page_content=merged_content, metadata=merged_meta))
            i += 2
        else:
            out.append(a)
            i += 1
    return out


@dataclass
class _DocLike:
    """Drop-in replacement for langchain Document used after Task/Solution
    merge — same .page_content / .metadata surface."""

    page_content: str
    metadata: dict


# ───────────────────────────── main entry point ──────────────────────────────


def chunk_md(path: Path, source_type: SourceType, slug: str) -> list[Section]:
    """Read a `.md` file and produce its list of Section objects.

    `slug` is the canonical doc identifier (filename without `.md`), validated
    upstream by `fill.manifest`. `source_type` comes from manifest.json.
    """
    raw = path.read_text(encoding="utf-8")
    fm = frontmatter.loads(raw)
    title = (fm.get("title") or slug).strip()
    body = fm.content

    # Always prepend synthetic H1 so the preamble paragraph(s) inherit a
    # heading_path. If the body already has an explicit H1 the splitter
    # nests it (still works, just produces one extra h1 in metadata).
    body_with_h1 = f"# {title}\n\n{body}"

    parts = _SPLITTER.split_text(body_with_h1)

    if not parts:
        # Defensive: should never happen with the synthetic H1 prepend, but
        # guarantees a non-empty list for files that are pure whitespace.
        return [
            _make_section(
                slug=slug,
                source_type=source_type,
                segments=[title],
                section_id=f"{slug}::_root",
                content=body.strip() or "(empty)",
            )
        ]

    if source_type == "how-to":
        parts = _group_task_solution_pairs(parts)

    sections: list[Section] = []
    section_id_use_count: dict[str, int] = {}
    task_counter = 0

    for part in parts:
        is_task_pair = part.metadata.get("_task_pair", False)
        if is_task_pair:
            task_counter += 1
            base_id = f"{slug}::task-{task_counter:03d}"
            # heading_path keeps the human breadcrumb to the H2 (e.g. "Example 1")
            segments = _heading_segments(part.metadata, default_h1=title)
        else:
            segments = _heading_segments(part.metadata, default_h1=title)
            base_id = f"{slug}::{_kebab_chain(part.metadata)}"

        # Duplicate-header dedup: second occurrence gets `::dup-1`, third `::dup-2`, ...
        count = section_id_use_count.get(base_id, 0)
        section_id_use_count[base_id] = count + 1
        if count > 0:
            section_id = f"{base_id}::dup-{count}"
        else:
            section_id = base_id

        # Normalize content once — Section storage strips, so size measurement
        # must too, else what we measure ≠ what we hash/upload (determinism).
        content_stripped = part.page_content.strip()

        # Dynamic prefix budget computed from the ACTUAL prefix tokens. The
        # 100-token static `PREFIX_TOKEN_BUDGET` is a default; long
        # `heading_path` values (deeply-nested H3 chains, long titles) require
        # tightening the secondary-split budget at runtime to keep the final
        # payload ≤ MAX_SECTION_TOKENS.
        prefix_tokens = count_tokens(_prefix(source_type, segments))
        if prefix_tokens > MAX_SECTION_TOKENS:
            raise RuntimeError(
                f"chunker: heading prefix alone is {prefix_tokens} tokens (> "
                f"MAX_SECTION_TOKENS={MAX_SECTION_TOKENS}). Heading path "
                f"{segments!r} is pathologically long; consider tightening "
                f"chunker.kebab_case / refactoring source doc."
            )

        if count_tokens(content_stripped) + prefix_tokens <= MAX_SECTION_TOKENS:
            sections.append(
                _make_section(
                    slug=slug,
                    source_type=source_type,
                    segments=segments,
                    section_id=section_id,
                    content=content_stripped,
                )
            )
            continue

        # Oversized — secondary_split with a per-section budget computed from
        # the actual prefix length (not the static PREFIX_TOKEN_BUDGET default).
        sub_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max(MAX_SECTION_TOKENS - prefix_tokens, 100),
            chunk_overlap=SECONDARY_OVERLAP_TOKENS,
            length_function=lambda s: len(TOKENIZER.encode(s)),
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        sub_contents = sub_splitter.split_text(content_stripped) or [content_stripped]

        for idx, sub in enumerate(sub_contents):
            sub_id = f"{section_id}::part-{idx:02d}"
            sub_section = _make_section(
                slug=slug,
                source_type=source_type,
                segments=segments,
                section_id=sub_id,
                content=sub,
            )
            # Hard invariant — any oversized sub here is a chunker bug.
            if count_tokens(sub_section.payload) > OPENAI_CHUNK_LIMIT:
                raise RuntimeError(
                    f"chunker: section {sub_id!r} payload "
                    f"{count_tokens(sub_section.payload)} tokens exceeds OpenAI cap "
                    f"{OPENAI_CHUNK_LIMIT}; secondary_split underconfigured"
                )
            sections.append(sub_section)

    return sections


def _prefix(source_type: SourceType, segments: list[str]) -> str:
    return f"# {source_type}: {' > '.join(segments)}\n\n"


def _make_payload(source_type: SourceType, segments: list[str], content: str) -> str:
    heading_path = " > ".join(segments)
    return f"# {source_type}: {heading_path}\n\n{content}"


def _make_section(
    slug: str,
    source_type: SourceType,
    segments: list[str],
    section_id: str,
    content: str,
) -> Section:
    heading_path = " > ".join(segments)
    section_name = segments[-1] if segments else "_root"
    return Section(
        slug=slug,
        section_id=section_id,
        section_name=section_name,
        heading_path=heading_path,
        source_url=build_source_url(slug),
        source_type=source_type,
        raw_content=content.strip(),
    )
