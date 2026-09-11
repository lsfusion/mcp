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
    """A tiny corpus: eight `how-to` chunks and one `paradigm`, each pointing
    along its own axis, so "closest to axis k" is exact and obvious."""
    dim = settings.EMBEDDING_DIMENSIONS
    rows = [
        {"section_id": f"Howto_a::s{i}", "sourceType": "how-to", "slug": "Howto_a",
         "heading_path": f"How-to: a > s{i}", "keywords": "", "text": f"text {i}"}
        for i in range(8)
    ] + [
        {"section_id": "Howto_b::s0", "sourceType": "how-to", "slug": "Howto_b",
         "heading_path": "How-to: b > s0", "keywords": "", "text": "another how-to"},
        {"section_id": "Paradigm_c::s0", "sourceType": "paradigm", "slug": "Paradigm_c",
         "heading_path": "Paradigm: c > s0", "keywords": "", "text": "paradigm text"},
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
    hits = li.search(_unit(settings.EMBEDDING_DIMENSIONS, 2), "how-to", 18)
    assert [h["section_id"] for h in hits][0] == "Howto_a::s2"
    assert hits[0]["score"] == pytest.approx(1.0)
    assert hits[0]["text"] == "text 2"


def test_a_branch_only_ever_returns_its_own_chunks(snapshot):
    hits = li.search(_unit(settings.EMBEDDING_DIMENSIONS, 0), "paradigm", 30)
    assert [h["section_id"] for h in hits] == ["Paradigm_c::s0"]


def test_exclusions_are_applied_before_the_cut_not_after(snapshot):
    """Dropping them from a finished top-k would return fewer chunks than the
    caller asked for — which is exactly what paging with `exclude_ids` needs
    not to happen."""
    hits = li.search(_unit(settings.EMBEDDING_DIMENSIONS, 2), "how-to", 12,
                     exclude_ids={"Howto_a::s2"})
    # "text N" is 6 characters, so 12 buys two — and the excluded one must not
    # be one of them, nor silently shorten the answer to one.
    assert len(hits) == 2 and "Howto_a::s2" not in [h["section_id"] for h in hits]


def test_no_snapshot_is_not_an_error_it_is_a_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(li, "SNAPSHOT_PATH", str(tmp_path / "absent.npz"))
    li.reset_for_tests()
    assert li.get() is None
    assert li.search(_unit(settings.EMBEDDING_DIMENSIONS, 0), "rules", 18) == []


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
    assert [h["section_id"] for h in li.search(_unit(dim, 0), "rules", 18)] == ["New::x"]


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


def _wire_tool(monkeypatch, snapshot_path, *, embed_axis=2, embed_raises=False):
    """Point retrieve_docs at the snapshot and stub the ONE network call the
    local path still makes — embedding the queries."""
    import tools.rag_retrieve as rr

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
    out = rr.retrieve_docs_tool("anything", type="how-to")
    assert out.docs[0].id == "Howto_a::s2"
    assert out.docs[0].score == pytest.approx(1.0)
    # The top guidance article is held back here exactly as it is on the store.
    assert "Rules::top" not in [d.id for d in out.docs]


def test_an_embeddings_outage_surfaces_instead_of_being_swallowed(snapshot, monkeypatch):
    """It used to degrade to the vector store. With the store gone the only
    honest move is to say so: an empty result set returned as a success is
    indistinguishable, to a model, from "no documentation exists"."""
    rr = _wire_tool(monkeypatch, snapshot, embed_raises=True)
    with pytest.raises(RuntimeError, match="embeddings are down"):
        rr.retrieve_docs_tool("anything", type="how-to")


def test_a_missing_snapshot_is_refused_not_answered_empty(tmp_path, monkeypatch):
    rr = _wire_tool(monkeypatch, tmp_path / "absent.npz")
    with pytest.raises(RuntimeError, match="nothing was searched"):
        rr.retrieve_docs_tool("anything", type="how-to")


# ─────────────────────────── several queries at once ─────────────────────────


def test_a_batch_answers_every_query_and_says_which(snapshot, monkeypatch):
    """Real traffic asks about unrelated things back to back — 60% of
    consecutive calls are on different topics — so a batch is the normal case,
    and the caller has to be able to tell the answers apart."""
    rr = _wire_tool(monkeypatch, snapshot)
    dim = settings.EMBEDDING_DIMENSIONS
    monkeypatch.setattr(rr, "_embed_queries",
                        lambda qs: [_unit(dim, 2), _unit(dim, 0)])
    out = rr.retrieve_docs_tool(["about s2", "about s0"], type="how-to")
    by_query = {}
    for d in out.docs:
        by_query.setdefault(d.query, []).append(d.id)
    assert set(by_query) == {"about s2", "about s0"}
    assert by_query["about s2"][0] == "Howto_a::s2"
    assert by_query["about s0"][0] == "Howto_a::s0"


def test_one_query_is_untouched_by_the_batch_path(snapshot, monkeypatch):
    """A single query must behave exactly as it did: same budget, and no
    `query` label on results that have nothing to be told apart from."""
    rr = _wire_tool(monkeypatch, snapshot)
    single = rr.retrieve_docs_tool("anything", type="how-to")
    as_list = rr.retrieve_docs_tool(["anything"], type="how-to")
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
    out = rr.retrieve_docs_tool(["weaker", "stronger"], type="how-to")
    ids = [d.id for d in out.docs]
    assert len(ids) == len(set(ids))
    assert [d.query for d in out.docs if d.id == "Howto_a::s2"] == ["stronger"]


def test_a_batch_shares_one_budget_instead_of_multiplying_it(snapshot, monkeypatch):
    """Otherwise batching becomes a way to buy context.

    The budget used to be a chunk count split across the queries; it is now
    characters split across the (query, branch) cells. What must not change is
    that two queries in one call cost no more than one query's worth of room.
    """
    rr = _wire_tool(monkeypatch, snapshot)
    dim = settings.EMBEDDING_DIMENSIONS
    monkeypatch.setattr(rr, "RESULT_MAX_CHARS", 400)
    monkeypatch.setattr(rr, "_embed_queries", lambda qs: [_unit(dim, 2)] * len(qs))
    out = rr.retrieve_docs_tool(["a", "b"], type="how-to")
    assert sum(len(d.text) for d in out.docs) <= 400


def test_too_many_queries_is_refused_not_truncated(snapshot, monkeypatch):
    """Split below one useful share and the call is worthless; say so instead
    of returning a token result per query."""
    rr = _wire_tool(monkeypatch, snapshot)
    monkeypatch.setattr(rr, "BATCH_MAX_QUERIES", 3)
    with pytest.raises(ValueError, match="at most 3 queries"):
        rr.retrieve_docs_tool(["a", "b", "c", "d"], type="how-to")


def test_an_empty_query_is_refused(snapshot, monkeypatch):
    rr = _wire_tool(monkeypatch, snapshot)
    for bad in ("", "   ", [], ["ok", ""]):
        with pytest.raises(ValueError, match="non-empty string"):
            rr.retrieve_docs_tool(bad, type="how-to")


def test_a_repeated_query_does_not_eat_its_own_budget(snapshot, monkeypatch):
    """["x", "x"] used to halve the share and then hand every chunk to the
    first copy — the caller got LESS than a plain "x" would have returned."""
    rr = _wire_tool(monkeypatch, snapshot)
    dim = settings.EMBEDDING_DIMENSIONS
    monkeypatch.setattr(rr, "_embed_queries", lambda qs: [_unit(dim, 2)] * len(qs))
    once = rr.retrieve_docs_tool("x", type="how-to")
    twice = rr.retrieve_docs_tool(["x", "x"], type="how-to")
    assert [d.id for d in twice.docs] == [d.id for d in once.docs]
    assert all(d.query is None for d in twice.docs)  # collapsed back to one query


def test_the_returned_list_is_ranked_by_score(snapshot, monkeypatch):
    """DocItem says the order means `score`; handing chunks out per query must
    not leave the list grouped by query instead."""
    rr = _wire_tool(monkeypatch, snapshot)
    dim = settings.EMBEDDING_DIMENSIONS
    monkeypatch.setattr(rr, "_embed_queries",
                        lambda qs: [_unit(dim, 0) * 0.4, _unit(dim, 2)])
    out = rr.retrieve_docs_tool(["weak", "strong"], type="how-to")
    scores = [d.score for d in out.docs]
    assert scores == sorted(scores, reverse=True)


def test_a_batch_logs_what_each_query_got(snapshot, monkeypatch):
    """One strong query hides a failed one behind the aggregate counts, so the
    per-query outcome has to be in the record."""
    rr = _wire_tool(monkeypatch, snapshot)
    dim = settings.EMBEDDING_DIMENSIONS
    events = _capture_emit(monkeypatch)
    monkeypatch.setattr(rr, "_embed_queries", lambda qs: [_unit(dim, 2), _unit(dim, 0)])
    rr.retrieve_docs_tool(["about s2", "about s0"], type="how-to")
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
    rr.retrieve_docs_tool(["a" * 20, "b" * 20, "c" * 20], type="how-to")
    assert sum(len(q) for q in events[0]["queries"]) <= 10


# --- naming ONE article ------------------------------------------------------
#
# A chunk answers the question it was ranked for and nothing else: the
# constraint it depends on, the case it omits and the table it points at are
# elsewhere in the same article and invisible from inside it. Naming the article
# is the way out; it is a paged traversal, not a whole-article read, because a
# reference article is not sized to arrive in one response.

def test_an_article_comes_back_in_document_order_with_no_scores(snapshot, monkeypatch):
    rr = _wire_tool(monkeypatch, snapshot)

    out = rr.retrieve_docs_tool(article="Howto_a")

    assert [d.id for d in out.docs] == [f"Howto_a::s{i}" for i in range(8)]
    assert [d.id for d in out.docs] == [f"Howto_a::s{i}" for i in range(8)]
    # Nothing to be similar to. Inventing a number here would make the order
    # look like a ranking it is not.
    assert all(d.score is None for d in out.docs)
    assert "`Howto_a`" in out.status
    assert "all 8 chunks" in out.status
    # The revision is not a response field: the only caller-facing use for it
    # is a traversal spanning a rebuild, and the status sentence says it there.
    assert "deadbeef" in out.status
    assert not hasattr(out, "revision")


def test_a_query_beside_an_article_searches_inside_it(snapshot, monkeypatch):
    rr = _wire_tool(monkeypatch, snapshot, embed_axis=5)

    out = rr.retrieve_docs_tool("anything", article="Howto_a")

    assert "`Howto_a`" in out.status
    assert {d.id for d in out.docs} <= {f"Howto_a::s{i}" for i in range(8)}
    assert out.docs[0].id == "Howto_a::s5"      # ranked, not walked
    assert all(d.score is not None for d in out.docs)


def test_the_article_total_ignores_what_the_caller_already_holds(snapshot, monkeypatch):
    # The total is the denominator of a traversal, so paging must not move it:
    # otherwise "I hold all of it" is unanswerable.
    rr = _wire_tool(monkeypatch, snapshot)

    out = rr.retrieve_docs_tool(article="Howto_a",
                                exclude_ids=["Howto_a::s0", "Howto_a::s1"])

    # The denominator is the whole article, not what this page could offer:
    # otherwise "how much of it do I have" has no stable answer while paging.
    assert "`Howto_a` 6 chunks here, none left" in out.status
    assert "nothing of it remains to fetch" in out.status


def test_document_order_stops_rather_than_leaving_a_hole(snapshot, monkeypatch):
    # Skipping an oversized chunk and carrying on would hand back sections
    # 1, 2 and 4 as if they were consecutive, with nothing saying otherwise.
    rr = _wire_tool(monkeypatch, snapshot)
    monkeypatch.setattr(rr, "RESULT_MAX_CHARS", 14)  # "text 0".."text 2" fit

    out = rr.retrieve_docs_tool(article="Howto_a")

    assert "6 chunks did not fit" in out.status


def test_a_chunk_id_and_a_link_destination_both_name_the_article(snapshot, monkeypatch):
    # The resolved name is not a field of its own: it is the prefix of every id
    # that comes back — which is what the caller pages with — and the status
    # sentence names it too.
    rr = _wire_tool(monkeypatch, snapshot)
    for name in ("Howto_a", "Howto_a::s3", "Howto_a.md", "../how-to/Howto_a.md"):
        out = rr.retrieve_docs_tool(article=name)
        assert {d.id.split("::")[0] for d in out.docs} == {"Howto_a"}
        assert "`Howto_a`" in out.status


def test_a_type_that_contradicts_the_article_is_an_error_not_an_empty_list(snapshot, monkeypatch):
    rr = _wire_tool(monkeypatch, snapshot)
    with pytest.raises(ValueError, match="already picks its branch"):
        rr.retrieve_docs_tool(article="Howto_a", type="paradigm")


def test_a_guidance_name_is_routed_not_answered_empty(snapshot, monkeypatch):
    # The corpus links to `Brief.md` and `Rules.md`, so a caller can reach these
    # names honestly. An empty success would teach that the subject is
    # undocumented.
    rr = _wire_tool(monkeypatch, snapshot)
    for name in ("Rules", "Brief", "Rules_view", "../rules/Rules_logic.md"):
        with pytest.raises(ValueError, match="lsfusion_get_guidance"):
            rr.retrieve_docs_tool(article=name)


def test_an_unknown_article_says_where_names_come_from(snapshot, monkeypatch):
    rr = _wire_tool(monkeypatch, snapshot)
    with pytest.raises(ValueError, match="NOT a finding"):
        rr.retrieve_docs_tool(article="Nosuch_article")


def test_neither_selector_is_refused_with_both_named(snapshot, monkeypatch):
    rr = _wire_tool(monkeypatch, snapshot)
    with pytest.raises(ValueError, match="`query`, `article`, or both"):
        rr.retrieve_docs_tool()


def test_a_traversal_says_in_words_whether_it_is_whole(snapshot, monkeypatch):
    # A JSON array has no tail to lose, so nine chunks of nineteen look exactly
    # like all nine there are. The difference has to be stated, not subtracted.
    rr = _wire_tool(monkeypatch, snapshot)

    whole = rr.retrieve_docs_tool(article="Howto_a")
    assert whole.status.startswith("ARTICLE_COMPLETE:")
    assert "all 8 chunks" in whole.status

    monkeypatch.setattr(rr, "RESULT_MAX_CHARS", 14)
    part = rr.retrieve_docs_tool(article="Howto_a")
    assert part.status.startswith("ARTICLE_PARTIAL:")
    assert "still to fetch" in part.status
    assert "`Howto_a` 2 of 8" in part.status
    assert "still to fetch" in part.status


def test_the_status_never_claims_to_know_what_the_caller_holds(snapshot, monkeypatch):
    # The server cannot see earlier pages, so every sentence is about THIS
    # response and one revision — never about the caller's collection.
    rr = _wire_tool(monkeypatch, snapshot)
    monkeypatch.setattr(rr, "RESULT_MAX_CHARS", 14)
    st = rr.retrieve_docs_tool(article="Howto_a").status
    assert "this response" in st and "deadbeef" in st
    assert "you hold" not in st.lower() and "you have not read" not in st.lower()


def test_an_exclusion_in_the_middle_still_ends_the_walk(snapshot, monkeypatch):
    # Exclusions can take chunks out of the middle. What is left is still
    # everything this call can fetch, so the walk is over — and the status says
    # that without claiming the whole article is in this response.
    rr = _wire_tool(monkeypatch, snapshot)
    out = rr.retrieve_docs_tool(article="Howto_a",
                                exclude_ids=["Howto_a::s1", "Howto_a::s2"])
    assert [d.id for d in out.docs] == [f"Howto_a::s{i}" for i in (0, 3, 4, 5, 6, 7)]
    assert "`Howto_a` 6 chunks here, none left" in out.status
    assert "all 8 chunks" not in out.status
    assert "To continue" not in out.status


def test_excluding_the_whole_article_is_an_answer_not_an_error(snapshot, monkeypatch):
    # A legitimate continuation can land here; calling it a failure would
    # confuse "nothing eligible remains" with "the lookup broke".
    rr = _wire_tool(monkeypatch, snapshot)
    out = rr.retrieve_docs_tool(article="Howto_a",
                                exclude_ids=[f"Howto_a::s{i}" for i in range(8)])
    assert out.docs == []
    assert out.status.startswith("ARTICLE_NONE:")
    assert "removed by `exclude_ids`" in out.status


def test_the_depth_is_characters_and_a_long_chunk_stops_the_walk(snapshot):
    """The count this took was only ever a number because the vector store's
    API took one; the ranking over the whole branch is already computed here."""
    # "text 0" .. "text 7", six characters each.
    assert len(li.search(_unit(settings.EMBEDDING_DIMENSIONS, 2), "how-to", 6)) == 1
    assert len(li.search(_unit(settings.EMBEDDING_DIMENSIONS, 2), "how-to", 12)) == 2
    assert len(li.search(_unit(settings.EMBEDDING_DIMENSIONS, 2), "how-to", 10_000)) == 9
    # The best chunk always comes back, even alone and over the depth.
    assert len(li.search(_unit(settings.EMBEDDING_DIMENSIONS, 2), "how-to", 1)) == 1


# --- the continuation protocol, driven to the end ----------------------------
#
# Not the first response: the SEQUENCE. A batch that finishes some articles and
# truncates others has to emit a call that names only the unfinished ones,
# carries every exclusion forward, advances every time, and eventually stops.
# Each of those can be wrong on its own while the first page looks perfect.

def _drive(rr, names, budget):
    """Follow the status's own continuation until it stops offering one."""
    import re
    excl: list[str] = []
    cur, pages = list(names), []
    for _ in range(50):
        out = rr.retrieve_docs_tool(article=cur, exclude_ids=excl or None)
        size = sum(len(d.text) for d in out.docs)
        assert size <= budget, f"page over budget: {size} > {budget}"
        pages.append({"asked": cur[:], "ids": [d.id for d in out.docs],
                      "status": out.status})
        if "To continue" not in out.status:
            return pages
        assert out.docs, "a non-terminal page returned nothing: the walk cannot advance"
        excl = excl + [d.id for d in out.docs]          # cumulative, never replaced
        cur = [x.strip(" '\"") for x in
               re.search(r"article=\[([^\]]*)\]", out.status).group(1).split(",")]
    raise AssertionError("the continuation never terminated")


def test_a_multi_article_walk_terminates_and_repeats_nothing(snapshot, monkeypatch):
    rr = _wire_tool(monkeypatch, snapshot)
    monkeypatch.setattr(rr, "RESULT_MAX_CHARS", 20)      # 6-char chunks: ~3 per page

    pages = _drive(rr, ["Howto_a", "Paradigm_c"], budget=20)

    seen = [i for p in pages for i in p["ids"]]
    assert len(seen) == len(set(seen)), "a chunk came back twice"
    assert set(seen) == {f"Howto_a::s{i}" for i in range(8)} | {"Paradigm_c::s0"}
    assert pages[-1]["status"].startswith("ARTICLE_COMPLETE:")
    # Each page asks only for what is still unfinished.
    assert all(set(p["asked"]) <= {"Howto_a", "Paradigm_c"} for p in pages)
    assert len(pages[-1]["asked"]) <= len(pages[0]["asked"])


def test_every_article_gets_a_foothold_however_small_the_share(snapshot, monkeypatch):
    # An article returned empty is indistinguishable from one that does not
    # exist — the same reason a query is never left with nothing. The foothold
    # is reserved before anything else, so a long article earlier in the list
    # cannot eat the only chunk a later one would have got.
    rr = _wire_tool(monkeypatch, snapshot)
    # "text 0" is 6 characters and "paradigm text" is 13: a budget of 19 is
    # exactly both footholds and nothing else, so an article that missed out
    # would have missed out to its neighbour rather than to the budget.
    monkeypatch.setattr(rr, "RESULT_MAX_CHARS", 19)

    out = rr.retrieve_docs_tool(article=["Howto_a", "Paradigm_c"])

    assert {d.id.split("::")[0] for d in out.docs} == {"Howto_a", "Paradigm_c"}


def test_an_unknown_name_does_not_throw_away_the_readable_ones(snapshot, monkeypatch):
    rr = _wire_tool(monkeypatch, snapshot)

    out = rr.retrieve_docs_tool(article=["Howto_a", "No_such_article"])

    assert {d.id.split("::")[0] for d in out.docs} == {"Howto_a"}
    assert "No article is named `No_such_article`" in out.status
    assert "will not resolve on a retry" in out.status


def test_all_names_unknown_is_an_error_not_an_empty_success(snapshot, monkeypatch):
    # With nothing readable there is no response to attach the report to, and
    # an empty list would read as "the subject is undocumented".
    rr = _wire_tool(monkeypatch, snapshot)
    with pytest.raises(ValueError, match="no article named"):
        rr.retrieve_docs_tool(article=["No_such", "Nor_this"])


def test_too_many_articles_says_why(snapshot, monkeypatch):
    rr = _wire_tool(monkeypatch, snapshot)
    monkeypatch.setattr(rr, "ARTICLE_MAX_NAMES", 2)
    with pytest.raises(ValueError, match="at most 2 articles"):
        rr.retrieve_docs_tool(article=["Howto_a", "Paradigm_c", "Howto_b"])


def test_a_search_inside_articles_never_claims_to_have_delivered_them(snapshot, monkeypatch):
    rr = _wire_tool(monkeypatch, snapshot, embed_axis=5)

    out = rr.retrieve_docs_tool("anything", article=["Howto_a", "Paradigm_c"])

    assert out.status.startswith(("COMPLETE:", "MORE:"))
    assert "search inside" in out.status
    assert "ARTICLE_COMPLETE" not in out.status


def test_an_article_too_big_to_start_is_a_dead_end_not_a_retry(snapshot, monkeypatch):
    # A continuation that offers an article whose next chunk cannot fit ANY
    # response is a loop dressed as progress. It has to be named as a dead end.
    rr = _wire_tool(monkeypatch, snapshot)
    monkeypatch.setattr(rr, "RESULT_MAX_CHARS", 8)   # "paradigm text" is 13

    out = rr.retrieve_docs_tool(article="Paradigm_c")

    assert out.docs == []
    assert out.status.startswith("ARTICLE_NONE:")
    assert "larger than one whole response" in out.status
    assert "To continue" not in out.status
