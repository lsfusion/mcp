"""The local dense backend: the snapshot, the search, and the fallback.

The point of these is not the cosine — it is that a bad or missing snapshot
degrades to the vector store instead of serving wrong answers, and that the
local branch keeps the same contract as its store twin (quota, top-article
exclusion, `exclude_ids` applied before the cut).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

import settings
from fill.snapshot import load, write
import tools.local_index as li


def _tamper(path: Path, edit) -> None:
    """Rewrite a snapshot's metadata in place — to prove the load-time checks
    catch a file damaged after it was written."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(bytes(z["meta"]).decode("utf-8"))
        vectors = z["vectors"]
    edit(meta)
    np.savez(path, meta=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
             vectors=vectors)


def _unit(dim: int, axis: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[axis] = 1.0
    return v


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    """A tiny corpus: four `rules` chunks and one `brief`, each pointing along
    its own axis, so "closest to axis k" is exact and obvious."""
    dim = settings.EMBEDDING_DIMENSIONS
    rows = [
        {"section_id": f"Rules_a::s{i}", "sourceType": "rules", "slug": "Rules_a",
         "heading_path": f"Rules: a > s{i}", "keywords": "", "text": f"text {i}"}
        for i in range(3)
    ] + [
        {"section_id": "Rules::top", "sourceType": "rules", "slug": "Rules",
         "heading_path": "Rules > top", "keywords": "", "text": "the top guidance article"},
        {"section_id": "Brief_b::s0", "sourceType": "brief", "slug": "Brief_b",
         "heading_path": "Brief: b > s0", "keywords": "", "text": "brief text"},
    ]
    vectors = np.stack([_unit(dim, i) for i in range(len(rows))])
    path = tmp_path / "corpus.npz"
    write(path, rows=rows, vectors=vectors,
          manifest={"embedding_model": settings.EMBEDDING_MODEL, "corpus_revision": "deadbeef",
                    "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    monkeypatch.setattr(li, "SNAPSHOT_PATH", str(path))
    li.reset_for_tests()
    yield path
    li.reset_for_tests()


def test_the_nearest_chunk_of_the_branch_comes_first(snapshot):
    hits = li.search(_unit(settings.EMBEDDING_DIMENSIONS, 2), "rules", 3)
    assert [h["section_id"] for h in hits][0] == "Rules_a::s2"
    assert hits[0]["score"] == pytest.approx(1.0)
    assert hits[0]["text"] == "text 2"


def test_a_branch_only_ever_returns_its_own_chunks(snapshot):
    hits = li.search(_unit(settings.EMBEDDING_DIMENSIONS, 0), "brief", 5)
    assert [h["section_id"] for h in hits] == ["Brief_b::s0"]


def test_exclusions_are_applied_before_the_cut_not_after(snapshot):
    """Dropping them from a finished top-k would return fewer chunks than the
    caller asked for — which is exactly what paging with `exclude_ids` needs
    not to happen."""
    hits = li.search(_unit(settings.EMBEDDING_DIMENSIONS, 2), "rules", 2,
                     exclude_ids={"Rules_a::s2"})
    assert len(hits) == 2 and "Rules_a::s2" not in [h["section_id"] for h in hits]


def test_the_top_guidance_article_can_be_held_back(snapshot):
    hits = li.search(_unit(settings.EMBEDDING_DIMENSIONS, 3), "rules", 4,
                     exclude_slugs={"Rules"})
    assert "Rules::top" not in [h["section_id"] for h in hits]


def test_no_snapshot_is_not_an_error_it_is_a_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(li, "SNAPSHOT_PATH", str(tmp_path / "absent.npz"))
    li.reset_for_tests()
    assert li.get() is None
    assert li.search(_unit(settings.EMBEDDING_DIMENSIONS, 0), "rules", 3) == []


def test_a_snapshot_from_another_model_is_refused(tmp_path, monkeypatch):
    """Not loaded, not partially trusted: query vectors from one model and
    document vectors from another do not fail, they just rank nonsense."""
    dim = settings.EMBEDDING_DIMENSIONS
    path = tmp_path / "corpus.npz"
    write(path, rows=[{"section_id": "X::y", "sourceType": "rules", "slug": "X",
                       "heading_path": "X > y", "keywords": "", "text": "t"}],
          vectors=np.stack([_unit(dim, 0)]),
          manifest={"embedding_model": "some-other-model", "corpus_revision": "x"})
    monkeypatch.setattr(li, "SNAPSHOT_PATH", str(path))
    li.reset_for_tests()
    assert li.get() is None


def test_a_truncated_snapshot_is_refused(tmp_path):
    """The failure that would otherwise be invisible: a file that loads but
    holds half the corpus."""
    dim = settings.EMBEDDING_DIMENSIONS
    path = tmp_path / "corpus.npz"
    write(path, rows=[{"section_id": "X::y", "sourceType": "rules", "slug": "X",
                       "heading_path": "X > y", "keywords": "", "text": "t"}],
          vectors=np.stack([_unit(dim, 0)]),
          manifest={"embedding_model": settings.EMBEDDING_MODEL, "corpus_revision": "x"})
    _tamper(path, lambda meta: meta["manifest"].update({"n_chunks": 99}))
    with pytest.raises(ValueError, match="truncated"):
        load(path, expect_model=settings.EMBEDDING_MODEL, expect_dimensions=dim)


def test_texts_that_do_not_match_the_manifest_are_refused(tmp_path):
    """The vector and the text it was computed from must travel together; if
    the texts were swapped, we would search by one corpus and quote another."""
    dim = settings.EMBEDDING_DIMENSIONS
    path = tmp_path / "corpus.npz"
    write(path, rows=[{"section_id": "X::y", "sourceType": "rules", "slug": "X",
                       "heading_path": "X > y", "keywords": "", "text": "original"}],
          vectors=np.stack([_unit(dim, 0)]),
          manifest={"embedding_model": settings.EMBEDDING_MODEL, "corpus_revision": "x"})
    _tamper(path, lambda meta: meta.update({"texts": ["tampered"]}))
    with pytest.raises(ValueError, match="digest"):
        load(path, expect_model=settings.EMBEDDING_MODEL, expect_dimensions=dim)


def test_an_old_snapshot_is_used_but_says_so(snapshot, monkeypatch, caplog):
    """Age is a warning, not a refusal: a stale index still answers, and taking
    retrieval down for a weekend over a docs change would be the worse failure.
    But it must be visible — a silently stale index quotes documentation that
    has moved on."""
    _tamper(snapshot, lambda meta: meta["manifest"].update(
        {"built_at": "2020-01-01T00:00:00+00:00"}))
    li.reset_for_tests()
    with caplog.at_level("WARNING"):
        assert li.get() is not None
    assert "days old" in caplog.text


def test_a_snapshot_with_no_build_time_is_flagged(snapshot, monkeypatch, caplog):
    _tamper(snapshot, lambda meta: meta["manifest"].pop("built_at", None))
    li.reset_for_tests()
    with caplog.at_level("WARNING"):
        assert li.get() is not None
    assert "no build time" in caplog.text


def test_an_empty_corpus_is_never_written(tmp_path):
    with pytest.raises(ValueError, match="empty corpus"):
        write(tmp_path / "c.npz", rows=[], vectors=np.zeros((0, 4), dtype=np.float32),
              manifest={"embedding_model": "m"})


# ─────────────────── the tool itself, on the local backend ───────────────────


def _wire_tool(monkeypatch, snapshot_path, *, backend="local", embed_axis=2, embed_raises=False):
    """Point retrieve_docs at the snapshot and stub the ONE network call the
    local path still makes — embedding the query."""
    import tools.rag_retrieve as rr

    monkeypatch.setattr(rr, "RETRIEVAL_BACKEND", backend)
    monkeypatch.setattr(li, "SNAPSHOT_PATH", str(snapshot_path))
    li.reset_for_tests()

    def fake_embed(query):
        if embed_raises:
            raise RuntimeError("embeddings are down")
        return _unit(settings.EMBEDDING_DIMENSIONS, embed_axis)

    monkeypatch.setattr(rr, "_embed_query", fake_embed)
    return rr


def test_the_switch_serves_the_call_from_the_snapshot(snapshot, monkeypatch):
    rr = _wire_tool(monkeypatch, snapshot)

    def must_not_be_called(**kwargs):
        raise AssertionError("the local backend must not call the vector store")

    monkeypatch.setattr(rr.client.vector_stores, "search", must_not_be_called)
    out = rr.retrieve_docs_tool("anything", type="rules")
    assert out.docs[0].id == "Rules_a::s2"
    assert out.docs[0].score == pytest.approx(1.0)
    # The top guidance article is held back here exactly as it is on the store.
    assert "Rules::top" not in [d.id for d in out.docs]


def test_an_embeddings_outage_falls_back_to_the_store(snapshot, monkeypatch):
    """The fallback is the whole reason the switch is safe to flip: the local
    path needs one network call, and when it fails the call still gets served."""
    rr = _wire_tool(monkeypatch, snapshot, embed_raises=True)
    called = []

    class _Resp:
        data = []

    def fake_search(**kwargs):
        called.append(kwargs)
        return _Resp()

    monkeypatch.setattr(rr.client.vector_stores, "search", fake_search)
    rr.retrieve_docs_tool("anything", type="rules")
    assert called, "the store was never asked"


def test_a_missing_snapshot_falls_back_to_the_store(tmp_path, monkeypatch):
    rr = _wire_tool(monkeypatch, tmp_path / "absent.npz")
    called = []

    class _Resp:
        data = []

    monkeypatch.setattr(rr.client.vector_stores, "search",
                        lambda **kw: (called.append(kw), _Resp())[1])
    rr.retrieve_docs_tool("anything", type="rules")
    assert called, "the store was never asked"


def test_the_backend_that_served_the_call_is_logged(snapshot, monkeypatch):
    """A canary that cannot tell which backend answered is not a canary."""
    rr = _wire_tool(monkeypatch, snapshot)
    events = []
    monkeypatch.setattr(rr, "emit",
                        lambda event, fields, *, stream, ok=True: events.append(fields))
    monkeypatch.setattr(rr.client.vector_stores, "search",
                        lambda **kw: (_ for _ in ()).throw(AssertionError("not the store")))
    rr.retrieve_docs_tool("anything", type="rules")
    assert events[0]["backend"] == "local"
