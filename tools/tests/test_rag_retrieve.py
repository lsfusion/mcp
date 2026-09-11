"""Tests for tools/rag_retrieve.py — branch coverage + type validation.

`_search_branch` is monkeypatched so no OpenAI call happens; we only
assert which branches get searched and how `type` is validated/mapped.
"""

from __future__ import annotations

import collections
import os
import types

# rag_retrieve constructs an OpenAI client at import time; the newer SDK refuses
# an empty key. Tests never hit the network (`_search_branch` is patched),
# so a dummy key is enough. Must be set before importing the module.
os.environ.setdefault("OPENAI_API_KEY", "test-key-unused")

import pytest

import tools.rag_retrieve as rr


@pytest.fixture(autouse=True)
def _index_present(monkeypatch):
    """The index is loaded and queries embed. Tests that care otherwise say so.

    Retrieval refuses to run without the index — an empty result set returned
    as a success is indistinguishable, to a model, from "no documentation
    exists" — so a unit test of the tool's own logic has to satisfy that first.
    """
    monkeypatch.setattr(rr.local_index, "get", lambda: types.SimpleNamespace(manifest={}))
    monkeypatch.setattr(rr, "_embed_queries", lambda qs: [None] * len(qs))


def _hit(source_type: str = "how-to", *, id: str = "X::sec", score: float = 0.9,
         text: str = "t") -> rr._Hit:
    return rr._Hit(
        id=id,
        source=f"documentation-{source_type}",
        text=text,
        score=score,
    )


def _record_calls(monkeypatch) -> list[dict]:
    """Record every `_search_branch` call (branch + exclusions)."""
    seen: list[dict] = []

    def fake_search(query_vector, source_type: str, exclude_ids=None):
        seen.append({"source_type": source_type, "exclude_ids": exclude_ids})
        return []

    monkeypatch.setattr(rr, "_search_branch", fake_search)
    return seen


def test_allowed_types_are_the_three_reference_branches():
    assert set(rr.ALLOWED_TYPES) == {"language", "paradigm", "how-to"}


def test_omitted_type_searches_every_searchable_branch(monkeypatch):
    calls = _record_calls(monkeypatch)
    rr.retrieve_docs_tool("anything")
    seen = [c["source_type"] for c in calls]
    assert set(seen) == {"language", "paradigm", "how-to"}
    assert len(seen) == 3


def test_default_types_is_allowed_types():
    assert set(rr.DEFAULT_TYPES) == set(rr.ALLOWED_TYPES)


def test_specific_type_searches_only_that_branch(monkeypatch):
    for t in rr.ALLOWED_TYPES:
        calls = _record_calls(monkeypatch)
        rr.retrieve_docs_tool("anything", type=t)
        assert [c["source_type"] for c in calls] == [t]


def test_a_guidance_branch_is_rejected_by_name(monkeypatch):
    # Not a silent empty result: an empty `docs: []` is indistinguishable, to a
    # model, from "no rules apply to this area". The error names the branches
    # that do exist here, and the tool description says where the other two went.
    for branch in ("rules", "brief"):
        calls = _record_calls(monkeypatch)
        with pytest.raises(ValueError):
            rr.retrieve_docs_tool("events", type=branch)
        assert calls == []


def test_the_searchable_branches_are_the_three_reference_ones():
    # Vacuity guard for everything below: if this list empties, the quota and
    # budget tests all pass for the wrong reason.
    assert set(rr.ALLOWED_TYPES) == {"language", "paradigm", "how-to"}
    assert rr.DEFAULT_TYPES == rr.ALLOWED_TYPES


# --- chunk ids -------------------------------------------------------------


class _FakeContent:
    type = "text"

    def __init__(self, text: str):
        self.text = text


class _FakeVSHit:
    def __init__(self, attributes: dict | None, score: float = 0.7):
        self.attributes = attributes
        self.score = score
        self.content = [_FakeContent("chunk body")]
        self.file_id = "vs_file_9"
        self.filename = "how-to/x.md"


class _FakeResp:
    def __init__(self, data):
        self.data = data


def _patch_vs_search(monkeypatch, resp, captured: dict | None = None):
    def fake_search(**kwargs):
        if captured is not None:
            captured.update(kwargs)
        return resp

    monkeypatch.setattr(rr.client.vector_stores, "search", fake_search)


# --- exclude_ids -----------------------------------------------------------


def test_exclude_ids_is_passed_to_every_branch(monkeypatch):
    calls = _record_calls(monkeypatch)
    rr.retrieve_docs_tool("q", exclude_ids=["A::x"])
    assert len(calls) == 3
    assert all(c["exclude_ids"] == ["A::x"] for c in calls)


# --- quotas ----------------------------------------------------------------


def test_every_requested_branch_is_searched_once_per_query(monkeypatch):
    # There is no per-branch count left to assert: depth is characters, and it
    # is the searcher's own business. What still matters is WHICH branches are
    # asked, and that a batch does not multiply the asking.
    for branch in rr.ALLOWED_TYPES:
        calls = _record_calls(monkeypatch)
        rr.retrieve_docs_tool("q", type=branch)
        assert [c["source_type"] for c in calls] == [branch]

    calls = _record_calls(monkeypatch)
    rr.retrieve_docs_tool("q")
    assert {c["source_type"] for c in calls} == set(rr.ALLOWED_TYPES)

    calls = _record_calls(monkeypatch)
    rr.retrieve_docs_tool(["one", "two"], type="how-to")
    assert len(calls) == 2  # one per query, not one per query per branch table


def test_the_branch_search_asks_the_index_for_the_whole_budget(monkeypatch):
    # The depth is the most any single cell could ever be granted, so a cell
    # that ends up with the whole budget still has candidates to spend it on.
    seen = {}
    monkeypatch.setattr(rr.local_index, "search",
                        lambda v, st, max_chars, exclude_ids=None: seen.setdefault("n", max_chars) and [])
    rr._search_branch(None, "how-to", None)
    assert seen["n"] == rr.RESULT_MAX_CHARS


# --- logging ---------------------------------------------------------------


def _capture_emit(monkeypatch) -> list[dict]:
    calls: list[dict] = []

    def fake_emit(event, fields, *, stream, ok=True):
        calls.append({"event": event, "fields": fields, "stream": stream, "ok": ok})

    monkeypatch.setattr(rr, "emit", fake_emit)
    return calls


def test_logs_retrieval_event_with_source_ids(monkeypatch):
    def fake_search(query_vector, source_type, exclude_ids=None):
        return [_hit(source_type, id=f"{source_type}::s")]

    monkeypatch.setattr(rr, "_search_branch", fake_search)
    calls = _capture_emit(monkeypatch)

    out = rr.retrieve_docs_tool("how to group", type="how-to")

    assert len(out.docs) == 1  # response carries DocItem (id/source/text/score) only
    assert not hasattr(out.docs[0], "file_id")
    assert len(calls) == 1
    c = calls[0]
    assert c["event"] == "retrieve_docs" and c["stream"] == "retrieval" and c["ok"] is True
    f = c["fields"]
    assert f["type"] == "how-to" and f["n_results"] == 1 and f["top_score"] == 0.9
    assert "query" in f and "text" not in f  # no chunk text logged
    assert f["results"][0] == {"rank": 1, "source": "documentation-how-to", "score": 0.9}


def test_the_log_counts_candidates_not_a_quota(monkeypatch):
    # `n_candidates` is what the branches returned before the budget chose; with
    # `n_results` and `n_omitted` beside it, it says whether the budget or the
    # corpus decided the answer. A quota count could say neither.
    calls = _capture_emit(monkeypatch)
    monkeypatch.setattr(rr, "_search_branch",
                        lambda v, s, e=None: [_hit(s, id=f"{s}{n}::x", text="t" * 10)
                                                 for n in range(4)])

    rr.retrieve_docs_tool("q", type="how-to")
    assert calls[-1]["fields"]["n_candidates"] == 4

    rr.retrieve_docs_tool("q")
    assert calls[-1]["fields"]["n_candidates"] == 12  # 3 branches x 4


def test_log_carries_exclusion_count_not_the_ids(monkeypatch):
    _record_calls(monkeypatch)
    calls = _capture_emit(monkeypatch)

    rr.retrieve_docs_tool("q", type="how-to", exclude_ids=["A::x", "B::y", "C::z"])
    f = calls[-1]["fields"]
    assert f["n_excluded"] == 3
    assert "A::x" not in repr(f)  # ids themselves never land in the log

    rr.retrieve_docs_tool("q", type="how-to")
    assert calls[-1]["fields"]["n_excluded"] == 0


def test_failed_call_also_logs_the_exclusion_count(monkeypatch):
    calls = _capture_emit(monkeypatch)
    with pytest.raises(ValueError):
        rr.retrieve_docs_tool("q", type="bogus", exclude_ids=["A::x"])
    f = calls[0]["fields"]
    assert calls[0]["ok"] is False
    assert f["n_excluded"] == 1
    assert f["n_candidates"] is None  # nothing was retrieved — type was rejected


def test_logs_failed_call_then_reraises(monkeypatch):
    calls = _capture_emit(monkeypatch)
    with pytest.raises(ValueError):
        rr.retrieve_docs_tool("anything", type="bogus")
    assert len(calls) == 1
    assert calls[0]["ok"] is False
    assert calls[0]["fields"]["error_class"] == "ValueError"


def test_query_is_capped_in_log(monkeypatch):
    monkeypatch.setattr(rr, "_search_branch",
                        lambda v, s, e=None: [])
    monkeypatch.setattr(rr, "QUERY_LOG_MAX_CHARS", 10)
    calls = _capture_emit(monkeypatch)
    rr.retrieve_docs_tool("x" * 100)
    assert len(calls[0]["fields"]["query"]) == 10


def test_logging_failure_never_breaks_success(monkeypatch):
    monkeypatch.setattr(rr, "_search_branch",
                        lambda v, s, e=None: [_hit(s, score=0.5)])

    def boom(*a, **k):
        raise RuntimeError("log boom")

    monkeypatch.setattr(rr, "emit", boom)  # _log_retrieval must swallow this
    out = rr.retrieve_docs_tool("q", type="how-to")
    assert len(out.docs) == 1  # retrieval still succeeds


def test_search_error_logs_once_and_reraises_same(monkeypatch):
    sentinel = RuntimeError("vs down")

    def boom_search(q, s, k, e=None):
        raise sentinel

    monkeypatch.setattr(rr, "_search_branch", boom_search)
    calls = _capture_emit(monkeypatch)
    with pytest.raises(RuntimeError) as ei:
        rr.retrieve_docs_tool("q", type="how-to")
    assert ei.value is sentinel  # original error propagates unchanged
    assert len(calls) == 1 and calls[0]["ok"] is False
    assert calls[0]["fields"]["error_class"] == "RuntimeError"


def test_error_message_is_capped(monkeypatch):
    monkeypatch.setattr(rr, "ERROR_LOG_MAX_CHARS", 5)
    monkeypatch.setattr(rr, "_search_branch",
                        lambda v, s, e=None: (_ for _ in ()).throw(RuntimeError("x" * 100)))
    calls = _capture_emit(monkeypatch)
    with pytest.raises(RuntimeError):
        rr.retrieve_docs_tool("q", type="how-to")
    assert len(calls[0]["fields"]["error_message"]) == 5


def test_error_message_never_leaks_excluded_ids(monkeypatch):
    # A provider error can echo the request back, filter values included. The
    # log has no field naming the excluded ids on purpose; the error string
    # must not become one through the back door.
    records: list[dict] = []
    monkeypatch.setattr(rr, "emit", lambda name, fields, **kw: records.append(fields))

    def boom(*a, **kw):
        raise RuntimeError("400 bad filter: {'nin': ['AGGR::syntax', 'GROUP::examples']}")

    monkeypatch.setattr(rr, "_search_branch", boom)
    with pytest.raises(RuntimeError):
        rr.retrieve_docs_tool("q", type="how-to",
                              exclude_ids=["AGGR::syntax", "GROUP::examples"])

    assert len(records) == 1
    blob = str(records[0])
    assert "AGGR::syntax" not in blob
    assert "GROUP::examples" not in blob
    assert "<excluded-id>" in records[0]["error_message"]
    assert records[0]["n_excluded"] == 2

# --- title rerank -----------------------------------------------------------


def _vshit(section_id: str, heading_path: str, score: float, slug: str = ""):
    return _FakeVSHit(
        {"section_id": section_id, "heading_path": heading_path, "slug": slug},
        score=score,
    )

# --- the SIZE budget ---------------------------------------------------------
#
# The chunk quota bounds how MANY chunks come back; it never bounded how much
# text, and chunk length spans 15x across the corpus. The same 24 chunks are
# ~27 KB of average text and ~183 KB of the largest, and callers were receiving
# 70 KB responses their client paged out to a file for them to parse.

def _serve(monkeypatch, texts: list[str]):
    """One branch answers with these texts, ranked in the order given."""
    hits = [_hit("how-to", id=f"A{i}::s", score=1.0 - i / 100, text=t)
            for i, t in enumerate(texts)]
    monkeypatch.setattr(rr, "_search_branch",
                        lambda v, s, e=None: list(hits))
    return hits


def test_the_size_budget_cuts_what_the_chunk_quota_does_not(monkeypatch):
    _serve(monkeypatch, ["a" * 100, "b" * 100, "c" * 100])
    monkeypatch.setattr(rr, "RESULT_MAX_CHARS", 250)

    out = rr.retrieve_docs_tool("q", "how-to")

    assert [d.id for d in out.docs] == ["A0::s", "A1::s"]
    assert out.status.startswith("MORE:") and "1 chunk did not fit" in out.status


def test_chunks_are_kept_whole_and_never_trimmed(monkeypatch):
    # Half a chunk answers nothing and cannot be paged for, so the budget drops
    # chunks rather than truncating their tails.
    _serve(monkeypatch, ["a" * 100, "b" * 100])
    monkeypatch.setattr(rr, "RESULT_MAX_CHARS", 150)

    out = rr.retrieve_docs_tool("q", "how-to")

    assert [d.text for d in out.docs] == ["a" * 100]
    assert out.status.startswith("MORE:") and "1 chunk did not fit" in out.status


def test_an_oversized_chunk_is_skipped_not_a_stop_sign(monkeypatch):
    # Ending the walk at the first chunk that does not fit would waste the rest
    # of the budget on nothing; the smaller ones behind it still fit.
    _serve(monkeypatch, ["a" * 50, "B" * 900, "c" * 50])
    monkeypatch.setattr(rr, "RESULT_MAX_CHARS", 200)

    out = rr.retrieve_docs_tool("q", "how-to")

    assert [d.id for d in out.docs] == ["A0::s", "A2::s"]
    assert out.status.startswith("MORE:") and "1 chunk did not fit" in out.status


def test_the_best_hit_comes_back_even_alone_and_over_budget(monkeypatch):
    # Returning nothing to a query that matched is the worse answer.
    _serve(monkeypatch, ["a" * 5000])
    monkeypatch.setattr(rr, "RESULT_MAX_CHARS", 100)

    out = rr.retrieve_docs_tool("q", "how-to")

    assert len(out.docs) == 1 and len(out.docs[0].text) == 5000
    assert out.status.startswith("COMPLETE:")


def test_omitted_is_zero_when_the_quota_was_the_only_limit(monkeypatch):
    # The distinction a silently truncated result destroys: a short list here
    # IS the whole answer, and paging will not lengthen it.
    _serve(monkeypatch, ["a" * 10, "b" * 10])
    monkeypatch.setattr(rr, "RESULT_MAX_CHARS", 36000)

    out = rr.retrieve_docs_tool("q", "how-to")

    assert len(out.docs) == 2
    assert out.status.startswith("COMPLETE:")


# --- the budget is split per (query, branch), and unused share flows on -------
#
# The old per-branch chunk quota did two jobs and was honest about neither: it
# never bounded size, and it silently guaranteed that all three branches were
# represented. The second job is real and is kept — in characters now.

def _serve_per_branch(monkeypatch, texts_by_branch: dict[str, list[str]]):
    def fake(query_vector, source_type, exclude_ids=None):
        return [_hit(source_type, id=f"{source_type}{i}::s", score=1.0 - i / 100, text=t)
                for i, t in enumerate(texts_by_branch.get(source_type, []))]
    monkeypatch.setattr(rr, "_search_branch", fake)


def _branches(out):
    return collections.Counter(d.source for d in out.docs)


def test_one_branch_cannot_take_the_whole_response(monkeypatch):
    # Every branch answers with more than a third of the budget. Ranked
    # globally, the highest-scoring branch would swallow everything and the
    # caller would never see the other two kinds of answer.
    _serve_per_branch(monkeypatch, {b: ["x" * 1000] * 20 for b in rr.ALLOWED_TYPES})
    monkeypatch.setattr(rr, "RESULT_MAX_CHARS", 9000)

    out = rr.retrieve_docs_tool("q")

    assert set(_branches(out)) == {f"documentation-{b}" for b in rr.ALLOWED_TYPES}
    assert set(_branches(out).values()) == {3}  # 9000 / 3 cells / 1000 each


def test_a_branch_with_nothing_releases_its_share(monkeypatch):
    # An even split that could not be borrowed against would waste a third of
    # the budget whenever one branch has no answer.
    _serve_per_branch(monkeypatch, {"paradigm": ["x" * 1000] * 20})
    monkeypatch.setattr(rr, "RESULT_MAX_CHARS", 9000)

    out = rr.retrieve_docs_tool("q")

    assert sum(len(d.text) for d in out.docs) == 9000
    assert set(_branches(out)) == {"documentation-paradigm"}


def test_one_query_of_a_batch_cannot_take_the_whole_response(monkeypatch):
    # The searcher no longer sees the query text — it gets the query VECTOR —
    # so the two queries are told apart by the vector the stub is handed.
    monkeypatch.setattr(rr, "_embed_queries", lambda qs: list(qs))

    def fake(query_vector, source_type, exclude_ids=None):
        greedy = query_vector == "greedy"
        n = 20 if greedy else 1
        return [_hit(source_type, id=f"{query_vector}-{source_type}{i}::s",
                     score=(1.0 if greedy else 0.1) - i / 100,
                     text="x" * 500)
                for i in range(n)]
    monkeypatch.setattr(rr, "_search_branch", fake)
    monkeypatch.setattr(rr, "RESULT_MAX_CHARS", 6000)

    out = rr.retrieve_docs_tool(["greedy", "quiet"], type="how-to")

    per_query = collections.Counter(d.query for d in out.docs)
    assert per_query["quiet"] == 1          # its one answer survives the greedy one
    assert per_query["greedy"] >= 1


# --- an unavailable index must not look like an empty corpus ------------------

def test_an_unloaded_index_is_refused_not_answered_empty(monkeypatch):
    # `docs: []` returned as a SUCCESS is the one answer this server must never
    # give: to a model it is indistinguishable from "no documentation exists on
    # this subject", and nothing downstream can tell. There is no second backend
    # to quietly cover it any more.
    monkeypatch.setattr(rr.local_index, "get", lambda: None)
    monkeypatch.setattr(rr, "_search_branch",
                        lambda *a, **k: pytest.fail("nothing may be searched without the index"))

    with pytest.raises(RuntimeError, match="nothing was searched"):
        rr.retrieve_docs_tool("q", type="how-to")


def test_an_embedding_failure_surfaces_instead_of_being_swallowed(monkeypatch):
    monkeypatch.setattr(rr, "_embed_queries",
                        lambda qs: (_ for _ in ()).throw(RuntimeError("embeddings down")))

    with pytest.raises(RuntimeError, match="embeddings down"):
        rr.retrieve_docs_tool("q", type="how-to")


# --- the relevance floor -----------------------------------------------------
#
# What the per-branch chunk count was really doing, in the unit it was really
# doing it in. Without it a search fills the whole size budget every time, and
# what it fills the tail with is noise: two RANDOM chunks of this corpus score
# 0.264 median and 0.552 at p99, and the tail the budget reached for sat at
# 0.501.

def test_the_tail_below_the_floor_is_dropped(monkeypatch):
    _serve(monkeypatch, ["a" * 10] * 4)
    hits = _serve(monkeypatch, [])
    monkeypatch.setattr(rr, "_search_branch", lambda v, s, e=None: [
        _hit("how-to", id=f"A{i}::s", score=sc, text="x" * 10)
        for i, sc in enumerate((0.60, 0.55, 0.51, 0.49, 0.20))])
    monkeypatch.setattr(rr, "SCORE_FLOOR_GAP", 0.10)

    out = rr.retrieve_docs_tool("q", type="how-to")

    # 0.49 and 0.20 sit more than 0.10 below the best score of 0.60.
    assert [d.score for d in out.docs] == [0.60, 0.55, 0.51]


def test_the_floor_is_relative_so_a_hard_query_still_gets_an_answer(monkeypatch):
    # A fixed threshold would answer a hard query with nothing and an easy one
    # with everything: the gap is measured DOWN from whatever this query found,
    # so a query whose best is 0.31 still gets an answer.
    monkeypatch.setattr(rr, "_search_branch", lambda v, s, e=None: [
        _hit("how-to", id=f"A{i}::s", score=sc, text="x" * 10)
        for i, sc in enumerate((0.31, 0.30, 0.10))])
    monkeypatch.setattr(rr, "SCORE_FLOOR_GAP", 0.10)

    out = rr.retrieve_docs_tool("q", type="how-to")
    assert [d.score for d in out.docs] == [0.31, 0.30]


def test_a_branch_whose_best_is_noise_is_dropped_whole(monkeypatch):
    def fake(v, source_type, e=None):
        sc = 0.60 if source_type == "language" else 0.20
        return [_hit(source_type, id=f"{source_type}::s", score=sc, text="x" * 10)]
    monkeypatch.setattr(rr, "_search_branch", fake)
    monkeypatch.setattr(rr, "SCORE_FLOOR_GAP", 0.10)

    out = rr.retrieve_docs_tool("q")
    assert [d.source for d in out.docs] == ["documentation-language"]


def test_one_query_never_deletes_another_query_s_only_answer(monkeypatch):
    # Two unrelated questions have unrelated score scales. A global floor would
    # let a strong match for one silently erase the other's answer.
    monkeypatch.setattr(rr, "_embed_queries", lambda qs: list(qs))
    monkeypatch.setattr(rr, "_search_branch", lambda v, s, e=None: [
        _hit(s, id=f"{v}-{s}::0", score=(0.60 if v == "easy" else 0.40), text="x" * 10)])
    monkeypatch.setattr(rr, "SCORE_FLOOR_GAP", 0.10)

    out = rr.retrieve_docs_tool(["easy", "hard"], type="how-to")
    assert {d.query for d in out.docs} == {"easy", "hard"}


def test_a_traversal_is_not_filtered_by_relevance(monkeypatch):
    # Walking an article has no query to be relevant to; a floor there would
    # silently delete the middle of a document.
    monkeypatch.setattr(rr.local_index, "known_article", lambda s: True)
    monkeypatch.setattr(rr.local_index, "article_branch", lambda s: "how-to")
    monkeypatch.setattr(rr.local_index, "article_chunks", lambda s, v, e: (
        [{"section_id": f"A::s{i}", "sourceType": "how-to", "slug": "A",
          "heading_path": "", "keywords": "", "text": "x" * 10,
          "score": None, "ordinal": i} for i in range(5)], 5))

    out = rr.retrieve_docs_tool(article="A")
    assert len(out.docs) == 5


def test_a_search_result_says_the_whole_article_can_be_had(monkeypatch):
    # The reported failure was an agent that could not get from "this chunk is
    # close" to "show me that article" and kept rephrasing instead. The
    # `article` parameter's own description explains the move — to someone who
    # has already decided to make it. This says it where they actually are.
    _serve(monkeypatch, ["a" * 10])
    out = rr.retrieve_docs_tool("q", type="how-to")
    assert "pass that chunk's `id` as `article`" in out.status


def test_the_chunk_id_says_what_it_is_good_for(monkeypatch):
    from tools.rag_retrieve import DocItem
    d = DocItem.model_fields["id"].description
    assert "`<article>::<section>`" in d
    assert "exclude_ids" in d and "as `article`" in d


def test_no_query_in_a_batch_leaves_with_nothing(monkeypatch):
    # An even split stops being fair once a share is smaller than a chunk: the
    # cells win or lose whole chunks, so a query whose best chunk merely
    # exceeds its share would be answered by the global pass or not at all —
    # and "not at all" is indistinguishable from "nothing matched".
    monkeypatch.setattr(rr, "_embed_queries", lambda qs: list(qs))
    monkeypatch.setattr(rr, "_search_branch", lambda v, s, e=None: [
        _hit(s, id=f"{v}-{s}::0", score=0.9, text="x" * 900)])
    monkeypatch.setattr(rr, "RESULT_MAX_CHARS", 1000)   # share is 1000/3 per cell

    out = rr.retrieve_docs_tool(["one", "two", "three"], type="how-to")

    assert {d.query for d in out.docs} == {"one", "two", "three"}


def test_too_many_queries_says_why_not_just_no(monkeypatch):
    monkeypatch.setattr(rr, "BATCH_MAX_QUERIES", 2)
    with pytest.raises(ValueError) as e:
        rr.retrieve_docs_tool(["a", "b", "c"])
    msg = str(e.value)
    assert "cannot both answer every query and stay inside one response" in msg
    assert "costs the same as the separate calls" in msg


def test_the_score_does_not_present_itself_as_confidence(monkeypatch):
    # "Similarity, higher = closer" invites the one reading the measurements
    # refuse: that a high number means a good answer. Among chunks agents later
    # reported on, the ones they called misleading did not score lower than the
    # ones they called helpful.
    from tools.rag_retrieve import DocItem
    d = DocItem.model_fields["score"].description
    assert "no absolute scale" in d
    assert "does not establish" in d
    assert "decided by reading it" in d
