"""The snapshot pipeline, end to end, with the embedding call stubbed.

This exists because of how removing the vector store could have gone wrong.
Nothing at request time touches the builder, so the whole suite and the running
server stay green against a snapshot that already exists — and the next
REBUILD is what fails, hours later, in a Jenkins job, with the stale snapshot
still being served as if nothing had happened. So the producer is exercised
here, not merely its cached output.
"""
from __future__ import annotations

import os
import types
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key-unused")

import settings
import tools.local_index as li
import tools.rag_build_snapshot as b


class _FakeEmbeddings:
    """One unit vector per input, along a rotating axis."""

    def __init__(self) -> None:
        self.calls = 0

    def create(self, model, input, dimensions=None, **kw):
        self.calls += 1
        n = settings.EMBEDDING_DIMENSIONS
        return types.SimpleNamespace(data=[
            types.SimpleNamespace(index=i, embedding=list(np.eye(n)[i % n]))
            for i in range(len(input))])


def _docs(root: Path) -> Path:
    for rel, body in (
        ("language/en/A_statement.md", "---\nslug: /A_statement\ntitle: 'A'\n---\n\n"
                                       "Lead.\n\n### Syntax\n\n`A`\n\n### Examples\n\n```lsf\nA;\n```\n"),
        ("paradigm/en/A_concept.md", "---\nslug: /A_concept\ntitle: 'C'\n---\n\nConcept lead.\n"),
        ("rules/en/Rules_logic.md", "---\nslug: /Rules_logic\ntitle: 'R'\n---\n\nNot indexed.\n"),
        ("language/ru/A_statement.md", "не индексируется"),
    ):
        f = root / "docs" / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body, encoding="utf-8")
    return root


def test_build_then_load_then_retrieve(tmp_path, monkeypatch):
    platform = _docs(tmp_path / "platform")
    out = tmp_path / "corpus.npz"
    client = types.SimpleNamespace(embeddings=_FakeEmbeddings())

    manifest = b.build(platform, out, client)

    # Built from the documentation SOURCE, and only the indexed English slice.
    assert manifest["n_chunks"] > 0
    assert set(manifest["chunks_per_branch"]) <= {"language", "paradigm", "how-to"}
    assert "rules" not in manifest["chunks_per_branch"]
    assert out.exists()

    monkeypatch.setattr(li, "SNAPSHOT_PATH", str(out))
    li.reset_for_tests()
    snap = li.get()
    assert snap is not None and len(snap) == manifest["n_chunks"]
    assert li.revision() == manifest["corpus_revision"]

    # And it is searchable, which is the only thing the server needs from it.
    hits = li.search(snap.vectors[0], snap.source_types[0], settings.RESULT_MAX_CHARS)
    assert hits and hits[0]["section_id"] == snap.section_ids[0]

    # The article path reads the same artifact.
    assert li.known_article("A_statement")
    rows, total = li.article_chunks("A_statement")
    assert total == len(rows) > 0
    assert [r["ordinal"] for r in rows] == list(range(total))


def test_the_builder_does_not_need_the_deleted_ingest_driver():
    # The coupling that would have made this deletion fail silently.
    import inspect
    src = inspect.getsource(b)
    assert "rag_ingest_docs" not in src
    with pytest.raises(ModuleNotFoundError):
        __import__("tools.rag_ingest_docs")
