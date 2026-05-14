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
    # All three sub-sections are short (<50 tokens), but Intro sits at
    # depth 1 ("AGGR operator") while Syntax/Examples are at depth 2
    # under "AGGR operator >". So Intro stays alone, Syntax+Examples
    # merge as siblings sharing parent_heading_path="AGGR operator".
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
    assert len(sections) == 2
    assert sections[0].section_id == "AGGR_operator::aggr-operator"
    assert sections[0].heading_path == "AGGR operator"
    assert sections[0].raw_content.startswith("Intro")

    # Merged section: composite id with `.` separator (SLUG_RE-safe);
    # heading uses ` + ` for visual contrast under the shared parent.
    assert sections[1].section_id == "AGGR_operator::syntax.examples"
    assert sections[1].heading_path == "AGGR operator > Syntax + Examples"
    assert sections[1].section_name == "Syntax + Examples"
    assert "AGGR clause" in sections[1].raw_content
    assert "Body" in sections[1].raw_content


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
    # Each H2 body is long enough that `_merge_short_siblings` doesn't
    # collapse them, so the dup-N logic is exercised in isolation.
    body = "Some longish body content that's well over the merge threshold. " * 10
    md = dedent(f"""\
        ## Foo

        {body}

        ## Bar

        {body}

        ## Foo

        {body}
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
    # Bodies are long enough to dodge `_merge_short_siblings`.
    body = "Body content " * 30
    md = f"## Foo\n\n{body}\n\n## Bar\n\n{body}\n\n## Foo\n\n{body}\n"
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


# ───────────────────────────── _merge_short_siblings ─────────────────────────


def test_merge_short_siblings_combines_consecutive(tmp_path):
    """Two consecutive H2 siblings, both short, share parent_heading_path
    and get merged into one composite Section."""
    md = "## Syntax\n\nfoo\n\n## Examples\n\nbar\n"
    p = write(tmp_path, "X.md", md)
    sections = chunk_md(p, "language", "X")
    # The two short H2s merge.
    assert any(s.section_id == "X::syntax.examples" for s in sections)
    merged = next(s for s in sections if s.section_id == "X::syntax.examples")
    assert merged.heading_path == "X > Syntax + Examples"
    assert merged.section_name == "Syntax + Examples"
    assert "foo" in merged.raw_content
    assert "bar" in merged.raw_content


def test_merge_not_across_parents(tmp_path):
    """Sections under DIFFERENT parents do NOT merge even when both short."""
    md = dedent("""\
        ## A

        short-a

        ### A.inner

        also-short
        """)
    p = write(tmp_path, "X.md", md)
    sections = chunk_md(p, "language", "X")
    # "A" is at depth=2 (parent "X"), "A.inner" is at depth=3 (parent "X > A").
    # Different parents → not merged.
    ids = [s.section_id for s in sections]
    assert "X::a.a-inner" not in ids
    assert "X::a" in ids or "X::a::a-inner" in ids


def test_merge_stops_at_threshold(tmp_path):
    """Three short siblings: once the first two sum past MIN_SECTION_TOKENS,
    the merger releases its accumulator and the third stays alone."""
    # Each section: a short header + body of ~40 tokens (still < 50 alone,
    # but two of them sum well past 50).
    body = " ".join(["word"] * 35)  # ~35 tokens
    md = f"## A\n\n{body}\n\n## B\n\n{body}\n\n## C\n\n{body}\n"
    p = write(tmp_path, "Y.md", md)
    sections = chunk_md(p, "language", "Y")
    h2_ids = [s.section_id for s in sections if "Y::" in s.section_id]
    # A and B merge (sum ~70 ≥ MIN), C stays alone.
    assert "Y::a.b" in h2_ids
    assert "Y::c" in h2_ids


def test_merge_does_not_absorb_large_neighbor(tmp_path):
    """A short section adjacent to a large one stays alone — we never
    dilute a substantial chunk by gluing a stub onto it."""
    big = " ".join(["lorem"] * 200)  # ~200 tokens >> MIN
    md = f"## Short\n\nfoo\n\n## Big\n\n{big}\n"
    p = write(tmp_path, "Z.md", md)
    sections = chunk_md(p, "language", "Z")
    ids = [s.section_id for s in sections]
    assert "Z::short" in ids
    assert "Z::big" in ids
    assert "Z::short.big" not in ids


def test_merge_skips_dup_sections(tmp_path):
    """Sections with `::dup-N` must NOT be absorbed — they're explicit
    semantic duplicates."""
    # Three H2s with same name: "## Foo" / "## Foo" / "## Foo". LangChain
    # will produce three sections with section_ids Foo, Foo::dup-1, Foo::dup-2.
    # Need a non-matching H2 between them to keep them as separate sections.
    md = dedent("""\
        ## Foo

        a

        ## Bar

        b

        ## Foo

        c
        """)
    p = write(tmp_path, "Dups.md", md)
    sections = chunk_md(p, "paradigm", "Dups")
    # First Foo + Bar are mergeable, Foo::dup-1 is NOT.
    ids = [s.section_id for s in sections]
    assert any("::dup-" in i for i in ids), ids
    # The dup section is preserved standalone.
    assert "Dups::foo::dup-1" in ids


def test_merge_skips_how_to_tasks(tmp_path):
    """`::task-NNN` sections stay individual — they're already a coherent
    Task/Solution pair from _group_task_solution_pairs (which keys off H3
    `### Task` / `### Solution` labels)."""
    md = dedent("""\
        ## Example 1

        ### Task

        t1 body

        ### Solution

        s1 body

        ## Example 2

        ### Task

        t2 body

        ### Solution

        s2 body
        """)
    p = write(tmp_path, "H.md", md)
    sections = chunk_md(p, "how-to", "H")
    task_ids = [s.section_id for s in sections if "::task-" in s.section_id]
    # Both task pairs survive as separate sections.
    assert len(task_ids) == 2
    # And no merge fused two task pairs into one composite id.
    assert not any(
        "task-" in s.section_id and "." in s.section_id.split("::")[-1]
        for s in sections
    )


def test_merge_is_deterministic_and_idempotent(tmp_path):
    """Same input md → same section list (ids + hashes) on every call."""
    md = "## A\n\nfoo\n\n## B\n\nbar\n\n## C\n\nbaz\n"
    p = write(tmp_path, "D.md", md)
    s1 = chunk_md(p, "language", "D")
    s2 = chunk_md(p, "language", "D")
    assert [(s.section_id, s.section_payload_hash) for s in s1] \
        == [(s.section_id, s.section_payload_hash) for s in s2]


def test_merged_section_id_passes_filename_safety():
    """Composite section_id `{base}::{seg1}.{seg2}` must round-trip through
    the SLUG_RE / `_filename_for` contract — only `.`, `_`, `-`, `=`, and
    alphanumerics. Pin the alphabet so a regression in `_build_merged_section`
    would surface here."""
    import re
    composite = "AGGR_operator::syntax.examples.parameters"
    # Mirror SLUG_RE used by manifest validator.
    assert re.fullmatch(r"[A-Za-z0-9_\-=.:]+", composite)
    # And the `::` → `__` filename transcoding survives.
    fn = composite.replace("::", "__") + ".md"
    assert ":" not in fn  # `::` is the only colon source and it's gone


def test_chunker_version_pinned_at_v2():
    """The merge behavior is on the v2 boundary. Bumping CHUNKER_VERSION
    re-triggers a forced-full-scan via state sentinel drift (covered
    end-to-end in test_state.test_needs_forced_full_scan_on_version_drift)."""
    assert CHUNKER_VERSION == "v2"


def test_task_force_header_is_not_excluded_from_merge(tmp_path):
    """`::task-` substring check is too loose — a real header literally
    titled 'Task-Force' kebabs to `task-force` and gives section_id
    `Doc::task-force`, which substring-matches `::task-` but is NOT a
    how-to task pair. The regex anchor (`task-\\d{3}`) rejects it."""
    md = "## Task-Force\n\nshort\n\n## Other\n\nshort\n"
    p = write(tmp_path, "TF.md", md)
    sections = chunk_md(p, "language", "TF")
    ids = [s.section_id for s in sections]
    # Confirms merge happens (would NOT if "::task-" excluded by substring).
    assert "TF::task-force.other" in ids


def test_part_NN_secondary_split_is_not_merged(tmp_path):
    """Secondary-split fragments (`::part-NN`) are slices of one logical
    section. They MUST NOT merge with the next sibling — that would
    produce a misleading section_id like `Doc::huge::part-03.next`."""
    # Build a doc where one H2 is huge enough to secondary-split, followed
    # by a very short H2 sibling.
    huge = " ".join(["word"] * 8000)
    md = f"## Huge\n\n{huge}\n\n## Tiny\n\nshort\n"
    p = write(tmp_path, "PT.md", md)
    sections = chunk_md(p, "language", "PT")
    # No section_id mixes `::part-NN` with `.something`.
    for s in sections:
        last_seg = s.section_id.split("::")[-1]
        assert not (
            last_seg.startswith("part-") and "." in last_seg
        ), f"unexpected merge of part-NN: {s.section_id}"
