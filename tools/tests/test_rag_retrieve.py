"""Tests for tools/rag_retrieve.py — branch coverage + type validation.

`_vs_search_for_source` is monkeypatched so no OpenAI call happens; we only
assert which branches get searched and how `type` is validated/mapped.
"""

from __future__ import annotations

import os

# rag_retrieve constructs an OpenAI client at import time; the newer SDK refuses
# an empty key. Tests never hit the network (`_vs_search_for_source` is patched),
# so a dummy key is enough. Must be set before importing the module.
os.environ.setdefault("OPENAI_API_KEY", "test-key-unused")

import pytest

import tools.rag_retrieve as rr


def _record_branches(monkeypatch) -> list[str]:
    seen: list[str] = []

    def fake_search(query: str, source_type: str, top_k: int):
        seen.append(source_type)
        return []

    monkeypatch.setattr(rr, "_vs_search_for_source", fake_search)
    return seen


def test_allowed_types_are_the_five_branches():
    assert set(rr.ALLOWED_TYPES) == {"language", "paradigm", "how-to", "brief", "rules"}


def test_omitted_type_searches_all_five_branches(monkeypatch):
    seen = _record_branches(monkeypatch)
    rr.retrieve_docs_tool("anything")
    assert set(seen) == set(rr.ALLOWED_TYPES)
    assert len(seen) == 5


def test_specific_type_searches_only_that_branch(monkeypatch):
    for t in rr.ALLOWED_TYPES:
        seen = _record_branches(monkeypatch)
        rr.retrieve_docs_tool("anything", type=t)
        assert seen == [t]


def test_bad_type_raises():
    with pytest.raises(ValueError):
        rr.retrieve_docs_tool("anything", type="bogus")


def test_every_allowed_type_has_a_top_k_mapping():
    for t in rr.ALLOWED_TYPES:
        assert t in rr._TYPE_TO_TOP_K
