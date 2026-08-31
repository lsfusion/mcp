"""The local dense backend: the snapshot, the search, and the fallback.

The point of these is not the cosine — it is that a bad or missing snapshot
degrades to the vector store instead of serving wrong answers, and that the
local branch keeps the same contract as its store twin (quota, top-article
exclusion, `exclude_ids` applied before the cut).
"""

from __future__ import annotations

import json
import os
import time
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
        for i in range(8)
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


def test_a_republished_snapshot_is_picked_up_without_a_restart(snapshot, monkeypatch):
    """The ingest publishes one on every docs change; restarting the server for
    each would drop every connected MCP session."""
    dim = settings.EMBEDDING_DIMENSIONS
    assert li.get().manifest["corpus_revision"] == "deadbeef"
    write(snapshot,
          rows=[{"section_id": "New::x", "sourceType": "rules", "slug": "New",
                 "heading_path": "New > x", "keywords": "", "text": "fresh"}],
          vectors=np.stack([_unit(dim, 0)]),
          manifest={"embedding_model": settings.EMBEDDING_MODEL, "corpus_revision": "cafe",
                    "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    os.utime(snapshot, (time.time() + 1, time.time() + 1))  # a fresh mtime, as a copy would have
    assert li.get().manifest["corpus_revision"] == "cafe"
    assert [h["section_id"] for h in li.search(_unit(dim, 0), "rules", 3)] == ["New::x"]


def test_a_bad_republish_keeps_the_snapshot_already_loaded(snapshot, monkeypatch, caplog):
    """A corpus one generation old beats none at all."""
    assert li.get() is not None
    _tamper(snapshot, lambda meta: meta["manifest"].update({"embedding_model": "other"}))
    os.utime(snapshot, (time.time() + 1, time.time() + 1))
    with caplog.at_level("ERROR"):
        assert li.get().manifest["corpus_revision"] == "deadbeef"
    assert "keeping the one already loaded" in caplog.text


def test_an_empty_corpus_is_never_written(tmp_path):
    with pytest.raises(ValueError, match="empty corpus"):
        write(tmp_path / "c.npz", rows=[], vectors=np.zeros((0, 4), dtype=np.float32),
              manifest={"embedding_model": "m"})


# ─────────────────── the tool itself, on the local backend ───────────────────


def _capture_emit(monkeypatch) -> list[dict]:
    calls: list[dict] = []
    monkeypatch.setattr(__import__("tools.rag_retrieve", fromlist=["x"]), "emit",
                        lambda event, fields, *, stream, ok=True: calls.append(fields))
    return calls


def _wire_tool(monkeypatch, snapshot_path, *, backend="local", embed_axis=2, embed_raises=False):
    """Point retrieve_docs at the snapshot and stub the ONE network call the
    local path still makes — embedding the queries."""
    import tools.rag_retrieve as rr

    monkeypatch.setattr(rr, "RETRIEVAL_BACKEND", backend)
    monkeypatch.setattr(li, "SNAPSHOT_PATH", str(snapshot_path))
    li.reset_for_tests()

    def fake_embed(queries):
        if embed_raises:
            raise RuntimeError("embeddings are down")
        return [_unit(settings.EMBEDDING_DIMENSIONS, embed_axis) for _ in queries]

    monkeypatch.setattr(rr, "_embed_queries", fake_embed)
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


# ─────────────────────────── several queries at once ─────────────────────────


def test_a_batch_answers_every_query_and_says_which(snapshot, monkeypatch):
    """Real traffic asks about unrelated things back to back — 60% of
    consecutive calls are on different topics — so a batch is the normal case,
    and the caller has to be able to tell the answers apart."""
    rr = _wire_tool(monkeypatch, snapshot)
    dim = settings.EMBEDDING_DIMENSIONS
    monkeypatch.setattr(rr, "_embed_queries",
                        lambda qs: [_unit(dim, 2), _unit(dim, 0)])
    out = rr.retrieve_docs_tool(["about s2", "about s0"], type="rules")
    by_query = {}
    for d in out.docs:
        by_query.setdefault(d.query, []).append(d.id)
    assert set(by_query) == {"about s2", "about s0"}
    assert by_query["about s2"][0] == "Rules_a::s2"
    assert by_query["about s0"][0] == "Rules_a::s0"


def test_one_query_is_untouched_by_the_batch_path(snapshot, monkeypatch):
    """A single query must behave exactly as it did: same budget, and no
    `query` label on results that have nothing to be told apart from."""
    rr = _wire_tool(monkeypatch, snapshot)
    single = rr.retrieve_docs_tool("anything", type="rules")
    as_list = rr.retrieve_docs_tool(["anything"], type="rules")
    assert [d.id for d in single.docs] == [d.id for d in as_list.docs]
    assert all(d.query is None for d in single.docs)


def test_a_chunk_answering_two_queries_goes_to_the_one_that_ranked_it_higher(snapshot, monkeypatch):
    """Separate calls cannot know a chunk is a repeat; one call can. And which
    query it is credited to is not "whichever was listed first" — it is the one
    the chunk actually answers better."""
    rr = _wire_tool(monkeypatch, snapshot)
    dim = settings.EMBEDDING_DIMENSIONS
    weak, strong = _unit(dim, 2) * 0.5, _unit(dim, 2)
    monkeypatch.setattr(rr, "_embed_queries", lambda qs: [weak, strong])
    out = rr.retrieve_docs_tool(["weaker", "stronger"], type="rules")
    ids = [d.id for d in out.docs]
    assert len(ids) == len(set(ids))
    assert [d.query for d in out.docs if d.id == "Rules_a::s2"] == ["stronger"]


def test_a_batch_shares_one_budget_instead_of_multiplying_it(snapshot, monkeypatch):
    """Otherwise batching becomes a way to buy context, and the quota stops
    meaning anything."""
    rr = _wire_tool(monkeypatch, snapshot)
    dim = settings.EMBEDDING_DIMENSIONS
    monkeypatch.setattr(rr, "BATCH_TOTAL_CAP", 4)
    monkeypatch.setattr(rr, "_embed_queries", lambda qs: [_unit(dim, 2)] * len(qs))
    out = rr.retrieve_docs_tool(["a", "b"], type="rules")
    assert len(out.docs) <= 4


def test_too_many_queries_is_refused_not_truncated(snapshot, monkeypatch):
    """Split below one useful share and the call is worthless; say so instead
    of returning a token result per query."""
    rr = _wire_tool(monkeypatch, snapshot)
    monkeypatch.setattr(rr, "BATCH_MAX_QUERIES", 3)
    with pytest.raises(ValueError, match="at most 3 queries"):
        rr.retrieve_docs_tool(["a", "b", "c", "d"], type="rules")


def test_an_empty_query_is_refused(snapshot, monkeypatch):
    rr = _wire_tool(monkeypatch, snapshot)
    for bad in ("", "   ", [], ["ok", ""]):
        with pytest.raises(ValueError, match="non-empty string"):
            rr.retrieve_docs_tool(bad, type="rules")


def test_a_repeated_query_does_not_eat_its_own_budget(snapshot, monkeypatch):
    """["x", "x"] used to halve the share and then hand every chunk to the
    first copy — the caller got LESS than a plain "x" would have returned."""
    rr = _wire_tool(monkeypatch, snapshot)
    dim = settings.EMBEDDING_DIMENSIONS
    monkeypatch.setattr(rr, "_embed_queries", lambda qs: [_unit(dim, 2)] * len(qs))
    once = rr.retrieve_docs_tool("x", type="rules")
    twice = rr.retrieve_docs_tool(["x", "x"], type="rules")
    assert [d.id for d in twice.docs] == [d.id for d in once.docs]
    assert all(d.query is None for d in twice.docs)  # collapsed back to one query


def test_the_returned_list_is_ranked_by_score(snapshot, monkeypatch):
    """DocItem says the order means `score`; handing chunks out per query must
    not leave the list grouped by query instead."""
    rr = _wire_tool(monkeypatch, snapshot)
    dim = settings.EMBEDDING_DIMENSIONS
    monkeypatch.setattr(rr, "_embed_queries",
                        lambda qs: [_unit(dim, 0) * 0.4, _unit(dim, 2)])
    out = rr.retrieve_docs_tool(["weak", "strong"], type="rules")
    scores = [d.score for d in out.docs]
    assert scores == sorted(scores, reverse=True)


def test_a_batch_logs_what_each_query_got(snapshot, monkeypatch):
    """One strong query hides a failed one behind the aggregate counts, so the
    per-query outcome has to be in the record."""
    rr = _wire_tool(monkeypatch, snapshot)
    dim = settings.EMBEDDING_DIMENSIONS
    events = _capture_emit(monkeypatch)
    monkeypatch.setattr(rr, "_embed_queries", lambda qs: [_unit(dim, 2), _unit(dim, 0)])
    rr.retrieve_docs_tool(["about s2", "about s0"], type="rules")
    f = events[0]
    assert f["queries"] == ["about s2", "about s0"]
    assert [s["query_index"] for s in f["query_stats"]] == [0, 1]
    assert all(s["n_results"] > 0 for s in f["query_stats"])
    assert {r["query_index"] for r in f["results"]} == {0, 1}


def test_the_batch_log_is_capped_as_a_whole(snapshot, monkeypatch):
    """Capping each query separately would let a batch write the cap times the
    batch size into one record."""
    rr = _wire_tool(monkeypatch, snapshot)
    dim = settings.EMBEDDING_DIMENSIONS
    events = _capture_emit(monkeypatch)
    monkeypatch.setattr(rr, "QUERY_LOG_MAX_CHARS", 10)
    monkeypatch.setattr(rr, "_embed_queries", lambda qs: [_unit(dim, 2)] * len(qs))
    rr.retrieve_docs_tool(["a" * 20, "b" * 20, "c" * 20], type="rules")
    assert sum(len(q) for q in events[0]["queries"]) <= 10
