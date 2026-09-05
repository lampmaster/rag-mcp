"""Parallel hybrid retrieval, RRF fusion and graceful degradation."""

import asyncio
import time

import pytest

from rag.chunk import make_chunk_id
from rag.fts import FTSUnavailableError, build_fts_index, fts_search
from rag.hybrid_search import (
    AllRetrieversFailedError,
    hybrid_search,
    hybrid_search_async,
    reciprocal_rank_fusion,
)
from rag.query_expansion import ExpandedQuery
from rag.search_result import SearchResult


def make_results(retriever, chunk_ids):
    return [
        SearchResult(
            chunk_id=cid,
            text=f"text of {cid}",
            source=cid.split("::")[0],
            rank=i,
            score=1.0 / i,
            retriever=retriever,
        )
        for i, cid in enumerate(chunk_ids, start=1)
    ]


# 10. RRF ordering -------------------------------------------------------------

def test_rrf_ordering_prefers_chunks_ranked_by_both_retrievers():
    vector = make_results("vector", ["a", "b", "c"])
    fts = make_results("fts", ["c", "d", "a"])

    fused = reciprocal_rank_fusion([vector, fts], k=60, top_k=5)
    ids = [r.chunk_id for r in fused]

    # c: 1/63 + 1/61, a: 1/61 + 1/63  -> tie, first-seen (a) wins
    # b: 1/62 only, d: 1/62 only
    assert ids[:2] == ["a", "c"]
    assert set(ids) == {"a", "b", "c", "d"}
    assert [r.rank for r in fused] == [1, 2, 3, 4]


def test_rrf_scores_follow_the_formula():
    vector = make_results("vector", ["x", "y"])
    fts = make_results("fts", ["y"])
    fused = reciprocal_rank_fusion([vector, fts], k=60, top_k=5)
    by_id = {r.chunk_id: r.score for r in fused}

    assert by_id["y"] == pytest.approx(1 / (60 + 2) + 1 / (60 + 1))
    assert by_id["x"] == pytest.approx(1 / (60 + 1))
    assert fused[0].chunk_id == "y", "ranks start at 1 and the shared hit wins"


def test_rrf_k_changes_the_weighting():
    vector = make_results("vector", ["a"])
    fused = reciprocal_rank_fusion([vector], k=10, top_k=1)
    assert fused[0].score == pytest.approx(1 / 11)


def test_rrf_respects_final_top_k():
    vector = make_results("vector", [f"v{i}" for i in range(10)])
    fts = make_results("fts", [f"f{i}" for i in range(10)])
    assert len(reciprocal_rank_fusion([vector, fts], top_k=5)) == 5


# 11. RRF deduplication by chunk_id --------------------------------------------

def test_rrf_deduplicates_by_chunk_id():
    vector = make_results("vector", ["dup", "a"])
    fts = make_results("fts", ["dup", "b"])

    fused = reciprocal_rank_fusion([vector, fts], top_k=10)
    ids = [r.chunk_id for r in fused]

    assert ids.count("dup") == 1
    assert len(ids) == len(set(ids)) == 3
    assert fused[0].metadata["rrf_ranks"] == {"vector": 1, "fts": 1}


def test_rrf_handles_empty_lists():
    assert reciprocal_rank_fusion([[], []]) == []
    fused = reciprocal_rank_fusion([[], make_results("fts", ["a"])])
    assert [r.chunk_id for r in fused] == ["a"]


# Parallel execution -----------------------------------------------------------

def test_retrievers_run_in_parallel():
    delay = 0.3

    def slow_vector(query, limit):
        time.sleep(delay)
        return make_results("vector", ["a"])

    def slow_fts(query, limit):
        time.sleep(delay)
        return make_results("fts", ["b"])

    started = time.perf_counter()
    outcome = hybrid_search(ExpandedQuery("q", [], []), slow_vector, slow_fts)
    elapsed = time.perf_counter() - started

    assert len(outcome.results) == 2
    assert elapsed < delay * 1.8, "FTS latency must not be added on top of vector"


def test_vector_gets_the_original_query_and_fts_gets_the_keywords():
    seen = {}

    def vector(query, limit):
        seen["vector"] = query
        return []

    def fts(query, limit):
        seen["fts"] = query
        return []

    expanded = ExpandedQuery("where is docker.md located", ["docker.md", "Dockerfile"], ["docker.md"])
    hybrid_search(expanded, vector, fts)

    assert seen["vector"] == "where is docker.md located"
    assert seen["fts"] == "where is docker.md located docker.md Dockerfile"


def test_candidate_counts_are_passed_to_the_retrievers():
    limits = {}

    def vector(query, limit):
        limits["vector"] = limit
        return []

    def fts(query, limit):
        limits["fts"] = limit
        return []

    hybrid_search(
        ExpandedQuery("q", [], []), vector, fts,
        vector_candidates=10, fts_candidates=10, top_k=5,
    )
    assert limits == {"vector": 10, "fts": 10}


# 12. graceful degradation when FTS fails --------------------------------------

def test_degrades_to_vector_when_fts_fails():
    def vector(query, limit):
        return make_results("vector", ["a", "b"])

    def broken_fts(query, limit):
        raise FTSUnavailableError("index missing")

    outcome = hybrid_search(ExpandedQuery("q", ["kw"], []), vector, broken_fts)

    assert [r.chunk_id for r in outcome.results] == ["a", "b"]
    assert outcome.fusion_used is False
    assert outcome.degraded is True
    assert isinstance(outcome.fts_error, FTSUnavailableError)
    assert outcome.vector_error is None


# 13. graceful degradation when Vector Search fails ----------------------------

def test_degrades_to_fts_when_vector_fails():
    def broken_vector(query, limit):
        raise RuntimeError("faiss index corrupted")

    def fts(query, limit):
        return make_results("fts", ["c", "d"])

    outcome = hybrid_search(ExpandedQuery("q", [], []), broken_vector, fts)

    assert [r.chunk_id for r in outcome.results] == ["c", "d"]
    assert [r.rank for r in outcome.results] == [1, 2]
    assert outcome.fusion_used is False
    assert isinstance(outcome.vector_error, RuntimeError)


def test_both_retrievers_failing_raises():
    def broken(query, limit):
        raise RuntimeError("down")

    with pytest.raises(AllRetrieversFailedError):
        hybrid_search(ExpandedQuery("q", [], []), broken, broken)


def test_hybrid_search_works_inside_a_running_event_loop():
    async def main():
        return hybrid_search(
            ExpandedQuery("q", [], []),
            lambda q, k: make_results("vector", ["a"]),
            lambda q, k: make_results("fts", ["b"]),
        )

    outcome = asyncio.run(main())
    assert len(outcome.results) == 2


# Exact technical term with a poor vector rank ---------------------------------

DOCS = {
    "docs/docker.md": "Local development uses docker-compose and the DOCKER_HOST variable.",
    "docs/security.md": (
        "Requests carry a Bearer credential. AUTH_TOKEN is issued by the identity "
        "service and signed with JWT_SECRET."
    ),
    "docs/database.md": "DB_POOL_SIZE controls the PostgreSQL pool behind DATABASE_URL.",
    "docs/kubernetes.md": "Workloads live in the namespace set by K8S_NAMESPACE.",
    "docs/mcp.md": "MCP (Model Context Protocol) exposes document tools to the model.",
}


@pytest.fixture
def fts_db(tmp_path):
    chunks = [
        {"chunk_id": make_chunk_id(s, 0), "source": s, "text": t, "chunk_index": 0}
        for s, t in DOCS.items()
    ]
    db_path = tmp_path / "fts.db"
    build_fts_index(chunks, db_path)
    return db_path


@pytest.mark.parametrize(
    "term, expected_doc",
    [
        ("AUTH_TOKEN", "docs/security.md"),
        ("docker.md", "docs/docker.md"),
        ("K8S_NAMESPACE", "docs/kubernetes.md"),
        ("DB_POOL_SIZE", "docs/database.md"),
    ],
)
def test_exact_term_with_bad_vector_rank_is_rescued_by_fts(fts_db, term, expected_doc):
    target = make_chunk_id(expected_doc, 0)

    # The embedding model does not recognise the rare term: the relevant chunk
    # only appears at rank 9 of the vector candidates, i.e. outside the top 5.
    noise = [make_chunk_id(s, 0) for s in DOCS if s != expected_doc]
    vector_ranking = [f"noise::{i}" for i in range(8)] + [target]

    def vector(query, limit):
        return make_results("vector", vector_ranking[:limit])

    def fts(query, limit):
        return fts_search(query, limit, fts_db)

    vector_only_top5 = [r.chunk_id for r in make_results("vector", vector_ranking)[:5]]
    assert target not in vector_only_top5, "precondition: vector-only misses it"

    outcome = hybrid_search(
        ExpandedQuery(f"where is {term} documented?", [term], []),
        vector, fts, vector_candidates=10, fts_candidates=10, top_k=5,
    )

    fused_ids = [r.chunk_id for r in outcome.results]
    assert outcome.fusion_used
    assert target in fused_ids, f"{term} must be rescued by FTS"
    assert fused_ids[0] == target, f"{term} should be ranked first after fusion"
    assert len(fused_ids) == len(set(fused_ids)), "no duplicates after fusion"
    assert noise  # documents other than the expected one exist in the index


def test_outcome_reports_latencies_and_candidate_counts():
    outcome = hybrid_search(
        ExpandedQuery("q", [], []),
        lambda q, k: make_results("vector", ["a", "b"]),
        lambda q, k: make_results("fts", ["b", "c"]),
    )
    assert len(outcome.vector_results) == 2
    assert len(outcome.fts_results) == 2
    assert outcome.vector_latency_ms >= 0
    assert outcome.fts_latency_ms >= 0


def test_async_entry_point_is_awaitable():
    outcome = asyncio.run(
        hybrid_search_async(
            ExpandedQuery("q", [], []),
            lambda q, k: make_results("vector", ["a"]),
            lambda q, k: make_results("fts", ["a"]),
        )
    )
    assert [r.chunk_id for r in outcome.results] == ["a"]
