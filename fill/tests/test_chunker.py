"""Tests for fill.chunker — markdown → list[Section]."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest

from fill.chunker import (
    MAX_SECTION_TOKENS,
    OPENAI_CHUNK_LIMIT,
    Section,
    chunk_md,
    count_tokens,
    kebab_case,
)
from fill.versions import (
    CHUNKER_VERSION,
    GLOSSARY_VERSION,
    PREFIX_VERSION,
    SOURCE_URL_VERSION,
)


# ───────────────────────────── tiny helpers ──────────────────────────────────


def write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


# ───────────────────────────── basic splits ──────────────────────────────────


def test_simple_split(tmp_path):
    md = dedent("""\
        ---
        title: AGGR operator
        ---
        Intro paragraph.

        ## Syntax

        ```
        AGGR clause
        ```

        ## Examples

        Body
        """)
    p = write(tmp_path, "AGGR_operator.md", md)
    sections = chunk_md(p, "language", "AGGR_operator")
    # 1 intro + 2 H2 sections
    assert len(sections) == 3
    assert sections[0].section_id == "AGGR_operator::aggr-operator"
    assert sections[0].heading_path == "AGGR operator"
    assert sections[0].section_name == "AGGR operator"
    assert sections[0].raw_content.startswith("Intro")

    assert sections[1].section_id == "AGGR_operator::syntax"
    assert sections[1].heading_path == "AGGR operator > Syntax"
    assert sections[1].section_name == "Syntax"

    assert sections[2].section_id == "AGGR_operator::examples"


def test_no_frontmatter_uses_slug_as_title(tmp_path):
    md = "Intro paragraph.\n\n## Syntax\n\nbody\n"
    p = write(tmp_path, "MyDoc.md", md)
    sections = chunk_md(p, "language", "MyDoc")
    assert sections[0].heading_path == "MyDoc"
    assert sections[1].heading_path == "MyDoc > Syntax"


def test_empty_body(tmp_path):
    p = write(tmp_path, "Empty.md", "---\ntitle: Empty\n---\n")
    sections = chunk_md(p, "paradigm", "Empty")
    assert len(sections) >= 1
    # The synthetic H1 + empty body produces at least one section with heading_path
    assert all(s.heading_path for s in sections)


def test_section_attributes_present(tmp_path):
    md = "## Foo\n\nbar\n"
    p = write(tmp_path, "X.md", md)
    sections = chunk_md(p, "paradigm", "X")
    s = next(s for s in sections if "foo" in s.section_id)
    assert s.slug == "X"
    assert s.source_type == "paradigm"
    assert s.source_url == "https://docs.lsfusion.org/X"
    assert s.heading_path == "X > Foo"
    assert s.section_name == "Foo"
    assert s.raw_content == "bar"


# ───────────────────────────── how-to grouping ───────────────────────────────


HOW_TO_MD = dedent("""\
    ---
    title: 'How-to: GROUP SUM'
    ---
    ## Example 1

    ### Task

    Count books per category.

    ### Solution

    ```lsf
    countBooks = GROUP SUM 1 BY category(Book b);
    ```

    ## Example 2

    ### Task

    Count tags.

    ### Solution

    ```lsf
    other code
    ```
    """)


def test_how_to_grouping_pairs(tmp_path):
    p = write(tmp_path, "How-to_GROUP_SUM.md", HOW_TO_MD)
    sections = chunk_md(p, "how-to", "How-to_GROUP_SUM")
    task_sections = [s for s in sections if "task-" in s.section_id]
    assert len(task_sections) == 2
    assert task_sections[0].section_id == "How-to_GROUP_SUM::task-001"
    assert task_sections[1].section_id == "How-to_GROUP_SUM::task-002"
    # Merged content includes both Task body and Solution body
    assert "Count books per category" in task_sections[0].raw_content
    assert "countBooks = GROUP SUM" in task_sections[0].raw_content


def test_how_to_grouping_orphan_task(tmp_path):
    # Task without following Solution → standalone
    md = dedent("""\
        ## Example

        ### Task

        Standalone task.
        """)
    p = write(tmp_path, "How-to_X.md", md)
    sections = chunk_md(p, "how-to", "How-to_X")
    # No task-NNN created — Task stays under regular kebab path
    assert not any("task-001" in s.section_id for s in sections)
    assert any(s.section_id.split("::")[-1] == "task" for s in sections)


def test_how_to_grouping_solution_without_task(tmp_path):
    md = dedent("""\
        ## Example

        ### Solution

        Lonely solution.
        """)
    p = write(tmp_path, "How-to_Y.md", md)
    sections = chunk_md(p, "how-to", "How-to_Y")
    # No task-NNN created
    assert not any("task-" in s.section_id for s in sections)


def test_how_to_grouping_only_for_how_to_sourcetype(tmp_path):
    # Same content but sourceType=paradigm — Task/Solution stay separate
    p = write(tmp_path, "Mixed.md", HOW_TO_MD)
    sections = chunk_md(p, "paradigm", "Mixed")
    assert not any("task-" in s.section_id for s in sections)


def test_how_to_ru_labels(tmp_path):
    md = dedent("""\
        ## Пример

        ### Задача

        Опиши задачу.

        ### Решение

        Опиши решение.
        """)
    p = write(tmp_path, "How-to_RU.md", md)
    sections = chunk_md(p, "how-to", "How-to_RU")
    assert any(s.section_id.endswith("::task-001") for s in sections)


# ───────────────────────────── section_id rules ──────────────────────────────


def test_duplicate_headers_get_dup_suffix(tmp_path):
    md = dedent("""\
        ## Foo

        first

        ## Bar

        between

        ## Foo

        second
        """)
    p = write(tmp_path, "Dup.md", md)
    sections = chunk_md(p, "paradigm", "Dup")
    foo_ids = [s.section_id for s in sections if s.section_id.startswith("Dup::foo")]
    assert foo_ids == ["Dup::foo", "Dup::foo::dup-1"]


def test_kebab_case():
    assert kebab_case("Foo Bar") == "foo-bar"
    assert kebab_case("CamelCase") == "camelcase"
    assert kebab_case("Spaces  Multiple") == "spaces-multiple"
    assert kebab_case("Special!#$") == "special"
    assert kebab_case("") == "section"


# ───────────────────────────── secondary split ───────────────────────────────


def test_oversized_section_is_split_into_parts(tmp_path):
    # Create a body whose single H2 section is > MAX_SECTION_TOKENS.
    # ~5 tokens per word * 10000 words ≈ 50000 tokens >> 2000.
    huge_body = " ".join(["lorem"] * 10000)
    md = f"## Huge\n\n{huge_body}\n"
    p = write(tmp_path, "Huge.md", md)
    sections = chunk_md(p, "paradigm", "Huge")
    huge_parts = [s for s in sections if "huge" in s.section_id]
    # Should split into multiple parts
    assert len(huge_parts) >= 2
    assert all("::part-" in s.section_id for s in huge_parts)
    # Each sub-section's full payload must fit in the OpenAI cap.
    for s in huge_parts:
        assert count_tokens(s.payload) <= OPENAI_CHUNK_LIMIT


def test_small_section_not_split(tmp_path):
    md = "## Tiny\n\nhi\n"
    p = write(tmp_path, "Tiny.md", md)
    sections = chunk_md(p, "paradigm", "Tiny")
    assert all("::part-" not in s.section_id for s in sections)


# ───────────────────────────── payload + hash ────────────────────────────────


def test_payload_shape(tmp_path):
    # No preamble body → LangChain doesn't emit a separate H1 section; the
    # H2 section inherits h1 metadata. We get one section_id "P::foo".
    md = "## Foo\n\nbody\n"
    p = write(tmp_path, "P.md", md)
    sections = chunk_md(p, "language", "P")
    foo = next(s for s in sections if s.section_id == "P::foo")
    assert foo.payload == "# language: P > Foo\n\nbody"


def test_payload_shape_with_preamble(tmp_path):
    # WITH preamble body → splitter emits both the H1 section (preamble) and the H2 section.
    md = "preamble paragraph.\n\n## Foo\n\nbody\n"
    p = write(tmp_path, "Q.md", md)
    sections = chunk_md(p, "language", "Q")
    section_ids = [s.section_id for s in sections]
    assert "Q::q" in section_ids        # kebab("Q") = "q"
    assert "Q::foo" in section_ids


def test_section_payload_hash_changes_with_content(tmp_path):
    p1 = write(tmp_path, "A.md", "## Foo\n\nv1\n")
    p2 = write(tmp_path, "A2.md", "## Foo\n\nv2\n")
    s1 = [s for s in chunk_md(p1, "language", "A") if "foo" in s.section_id][0]
    s2 = [s for s in chunk_md(p2, "language", "A") if "foo" in s.section_id][0]
    assert s1.section_payload_hash != s2.section_payload_hash


def test_section_payload_hash_changes_with_sourcetype(tmp_path):
    p = write(tmp_path, "B.md", "## Foo\n\nx\n")
    s_lang = [s for s in chunk_md(p, "language", "B") if "foo" in s.section_id][0]
    s_para = [s for s in chunk_md(p, "paradigm", "B") if "foo" in s.section_id][0]
    assert s_lang.section_payload_hash != s_para.section_payload_hash


def test_section_payload_hash_includes_all_versions(tmp_path):
    # Touch every version constant + section_id in the hashed payload to confirm wiring.
    p = write(tmp_path, "C.md", "## Foo\n\nx\n")
    s = [x for x in chunk_md(p, "language", "C") if "foo" in x.section_id][0]
    payload_text = json.dumps(
        {
            "section_id": s.section_id,
            "content": s.raw_content,
            "sourceType": s.source_type,
            "heading_path": s.heading_path,
            "chunker_version": CHUNKER_VERSION,
            "glossary_version": GLOSSARY_VERSION,
            "prefix_version": PREFIX_VERSION,
            "source_url_version": SOURCE_URL_VERSION,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    import hashlib
    expected = "sha256:" + hashlib.sha256(payload_text.encode()).hexdigest()
    assert s.section_payload_hash == expected


def test_section_payload_hash_stable_across_calls(tmp_path):
    p = write(tmp_path, "D.md", "## Foo\n\nbody\n")
    s1 = [s for s in chunk_md(p, "language", "D") if "foo" in s.section_id][0]
    s2 = [s for s in chunk_md(p, "language", "D") if "foo" in s.section_id][0]
    assert s1.section_payload_hash == s2.section_payload_hash


# ───────────────────────────── code fences ───────────────────────────────────


def test_code_fence_with_h3_inside_does_not_split(tmp_path):
    # A fenced code block contains `### header inside`. LangChain's
    # MarkdownHeaderTextSplitter should treat the fence as opaque.
    md = dedent("""\
        ## Real Header

        ```python
        ### this is a comment, not a header
        x = 1
        ```

        body after fence
        """)
    p = write(tmp_path, "Code.md", md)
    sections = chunk_md(p, "paradigm", "Code")
    # We should NOT get a section called `this-is-a-comment-not-a-header`.
    assert not any("comment" in s.section_id for s in sections)
    # The fenced code stays inside the "Real Header" section.
    real = next(s for s in sections if "real-header" in s.section_id)
    assert "### this is a comment" in real.raw_content


# ───────────────────────────── real-data invariant ───────────────────────────


REAL_DOCS = Path(__file__).resolve().parents[3] / "platform" / "docs" / "en"


def _real_md_samples():
    if not REAL_DOCS.is_dir():
        return []
    # Pick a representative sample to keep test runtime bounded.
    interesting = [
        "AGGR_operator.md",
        "Aggregations.md",
        "How-to_GROUP_SUM.md",
        "=_statement.md",
        "IF_..._THEN_operator.md",
    ]
    out = []
    for name in interesting:
        p = REAL_DOCS / name
        if p.is_file():
            out.append(p)
    return out


@pytest.mark.parametrize("path", _real_md_samples(),
                         ids=lambda p: p.name if hasattr(p, "name") else str(p))
def test_invariant_no_payload_exceeds_safe_limit(path):
    """Real docs: every Section's payload fits in OPENAI_CHUNK_LIMIT tokens."""
    slug = path.stem
    # Use paradigm as a neutral source_type; the invariant must hold for any.
    sections = chunk_md(path, "paradigm", slug)
    assert sections, f"empty sections from {path}"
    for s in sections:
        n = count_tokens(s.payload)
        assert n <= OPENAI_CHUNK_LIMIT, (
            f"{path.name} section {s.section_id} has {n} tokens "
            f"(> {OPENAI_CHUNK_LIMIT})"
        )


def test_how_to_merged_heading_path_stops_at_h2(tmp_path):
    # Task/Solution merged section should NOT have h3 in heading_path; it points
    # to the H2 only (e.g. "Title > Example 1"), since the merged content
    # contains both Task and Solution.
    p = write(tmp_path, "How-to_GROUP_SUM.md", HOW_TO_MD)
    sections = chunk_md(p, "how-to", "How-to_GROUP_SUM")
    task_sec = next(s for s in sections if s.section_id.endswith("::task-001"))
    assert "Task" not in task_sec.heading_path
    assert "Solution" not in task_sec.heading_path
    assert "Example 1" in task_sec.heading_path


def test_section_payload_hash_distinguishes_dup_ids(tmp_path):
    # Two duplicate-header siblings with identical raw_content + heading_path
    # produce DIFFERENT hashes (section_id is in the hash payload).
    # LangChain merges consecutive same-h2 sections, so separate them with
    # a different h2 in between.
    md = "## Foo\n\nbody\n\n## Bar\n\nsep\n\n## Foo\n\nbody\n"
    p = write(tmp_path, "Dup2.md", md)
    sections = chunk_md(p, "paradigm", "Dup2")
    foos = [s for s in sections if s.section_id.startswith("Dup2::foo")]
    assert len(foos) == 2, [s.section_id for s in sections]
    assert foos[0].section_id != foos[1].section_id
    assert foos[0].raw_content == foos[1].raw_content
    assert foos[0].section_payload_hash != foos[1].section_payload_hash


def test_section_payload_hash_changes_with_version_constants(tmp_path, monkeypatch):
    p = write(tmp_path, "V.md", "## Foo\n\nx\n")
    s_before = [s for s in chunk_md(p, "language", "V") if "foo" in s.section_id][0]
    hash_before = s_before.section_payload_hash
    monkeypatch.setattr("fill.chunker.CHUNKER_VERSION", "v999")
    s_after = [s for s in chunk_md(p, "language", "V") if "foo" in s.section_id][0]
    assert s_after.section_payload_hash != hash_before


def test_oversized_with_long_heading_path_still_fits(tmp_path):
    # Edge: long synthetic title eats into the prefix budget. Dynamic-budget
    # secondary_split must still produce all-fit chunks.
    title = "X" * 300   # long title — produces a long heading_path
    huge_body = " ".join(["lorem"] * 8000)
    md = f"---\ntitle: {title}\n---\n## Sec\n\n{huge_body}\n"
    p = write(tmp_path, "LongHead.md", md)
    sections = chunk_md(p, "paradigm", "LongHead")
    for s in sections:
        assert count_tokens(s.payload) <= OPENAI_CHUNK_LIMIT


def test_real_how_to_doc_produces_task_pairs():
    p = REAL_DOCS / "How-to_GROUP_SUM.md"
    if not p.is_file():
        pytest.skip("real doc not present in this checkout")
    sections = chunk_md(p, "how-to", "How-to_GROUP_SUM")
    task_ids = [s.section_id for s in sections if "::task-" in s.section_id]
    # The real doc has at least 3 Task/Solution pairs
    assert len(task_ids) >= 3, f"got task_ids={task_ids}"
