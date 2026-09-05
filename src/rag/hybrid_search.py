"""Hybrid retrieval: parallel Vector + FTS search fused with RRF.

           ┌── FAISS vector search ──┐
Query ─────┤                         ├── Reciprocal Rank Fusion ── Top-K
           └── SQLite FTS5 search ───┘

The retrievers are injected as plain callables, which keeps this module free
of FAISS/sqlite imports and makes graceful degradation easy to test.
"""

import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FINAL_TOP_K, FTS_CANDIDATES, RRF_K, VECTOR_CANDIDATES
from rag.query_expansion import ExpandedQuery
from rag.search_result import SearchResult

logger = logging.getLogger(__name__)


class AllRetrieversFailedError(Exception):
    """Raised when neither the vector nor the FTS retriever produced results."""


@dataclass
class HybridSearchOutcome:
    """Fused results plus diagnostics (used by logging and the benchmark)."""

    results: list[SearchResult]
    vector_results: list[SearchResult]
    fts_results: list[SearchResult]
    vector_error: Exception | None = None
    fts_error: Exception | None = None
    vector_latency_ms: float = 0.0
    fts_latency_ms: float = 0.0
    fusion_used: bool = False

    @property
    def degraded(self) -> bool:
        return self.vector_error is not None or self.fts_error is not None


# --------------------------------------------------------------------------
# Reciprocal Rank Fusion
# --------------------------------------------------------------------------

def reciprocal_rank_fusion(
    result_lists,
    k: int = None,
    top_k: int = None,
) -> list[SearchResult]:
    """Fuse ranked lists:  RRF(d) = Σ 1 / (k + rank_i(d)),  ranks start at 1.

    Results are merged and deduplicated by the stable `chunk_id`; raw FAISS
    similarities and BM25 scores are never combined directly (different scales).
    """
    k = RRF_K if k is None else k
    top_k = FINAL_TOP_K if top_k is None else top_k

    scores: dict[str, float] = {}
    best: dict[str, SearchResult] = {}
    order: list[str] = []
    contributions: dict[str, dict[str, int]] = {}

    for results in result_lists:
        for position, result in enumerate(results or [], start=1):
            # Trust an explicit rank when the retriever provided one.
            rank = result.rank if getattr(result, "rank", None) else position
            chunk_id = result.chunk_id
            if chunk_id not in scores:
                scores[chunk_id] = 0.0
                best[chunk_id] = result
                order.append(chunk_id)
                contributions[chunk_id] = {}
            scores[chunk_id] += 1.0 / (k + rank)
            retriever = result.retriever or f"retriever{len(contributions[chunk_id])}"
            contributions[chunk_id][retriever] = rank
            # Prefer a representation that actually carries text.
            if not best[chunk_id].text and result.text:
                best[chunk_id] = result

    first_seen = {cid: i for i, cid in enumerate(order)}
    fused = []
    for position, chunk_id in enumerate(
        sorted(order, key=lambda cid: (-scores[cid], first_seen[cid])), start=1
    ):
        source = best[chunk_id]
        fused.append(
            SearchResult(
                chunk_id=chunk_id,
                text=source.text,
                source=source.source,
                rank=position,
                score=scores[chunk_id],
                retriever="rrf",
                metadata={**source.metadata, "rrf_ranks": contributions[chunk_id]},
            )
        )

    return fused[:top_k]


# --------------------------------------------------------------------------
# Parallel retrieval
# --------------------------------------------------------------------------

async def _timed(fn, *args):
    """Run a blocking retriever in a worker thread, measuring latency."""
    started = time.perf_counter()
    try:
        results = await asyncio.to_thread(fn, *args)
        return results, None, (time.perf_counter() - started) * 1000
    except Exception as exc:
        return None, exc, (time.perf_counter() - started) * 1000


async def hybrid_search_async(
    expanded: ExpandedQuery,
    vector_fn,
    fts_fn,
    vector_candidates: int = None,
    fts_candidates: int = None,
    top_k: int = None,
    rrf_k: int = None,
) -> HybridSearchOutcome:
    """Run both retrievers concurrently and fuse what came back.

    Search query strategy:
        vector -> the ORIGINAL user query (keywords would distort the embedding)
        fts    -> the original user query + generated keywords
    """
    vector_candidates = VECTOR_CANDIDATES if vector_candidates is None else vector_candidates
    fts_candidates = FTS_CANDIDATES if fts_candidates is None else fts_candidates
    top_k = FINAL_TOP_K if top_k is None else top_k
    rrf_k = RRF_K if rrf_k is None else rrf_k

    original = expanded.original
    fts_query = " ".join([original, *expanded.keywords]).strip()

    # Both retrievers start at the same time - FTS latency must not be added
    # on top of vector latency.
    vector_task = _timed(vector_fn, original, vector_candidates)
    fts_task = _timed(fts_fn, fts_query, fts_candidates)
    (vector_results, vector_error, vector_ms), (fts_results, fts_error, fts_ms) = (
        await asyncio.gather(vector_task, fts_task)
    )

    if vector_error is not None:
        logger.warning("Vector search failed (%s) - degrading to FTS", vector_error)
    else:
        logger.info(
            "Vector search: %s candidates in %.1f ms",
            len(vector_results), vector_ms,
        )
    if fts_error is not None:
        logger.warning("FTS search failed (%s) - degrading to vector", fts_error)
    else:
        logger.info("FTS retrieval: %s candidates in %.1f ms", len(fts_results), fts_ms)

    outcome = HybridSearchOutcome(
        results=[],
        vector_results=vector_results or [],
        fts_results=fts_results or [],
        vector_error=vector_error,
        fts_error=fts_error,
        vector_latency_ms=vector_ms,
        fts_latency_ms=fts_ms,
    )

    if vector_error is not None and fts_error is not None:
        raise AllRetrieversFailedError(
            f"vector: {vector_error}; fts: {fts_error}"
        )

    if vector_error is None and fts_error is None:
        outcome.results = reciprocal_rank_fusion(
            [outcome.vector_results, outcome.fts_results], k=rrf_k, top_k=top_k
        )
        outcome.fusion_used = True
    elif vector_error is None:
        # FTS is down - keep the vector ranking as-is.
        outcome.results = _renumber(outcome.vector_results, top_k)
    else:
        # Vector search is down - keep the FTS ranking as-is.
        outcome.results = _renumber(outcome.fts_results, top_k)

    logger.info(
        "Hybrid results (fusion=%s): %s",
        outcome.fusion_used, [r.chunk_id for r in outcome.results],
    )
    return outcome


def _renumber(results, top_k: int) -> list[SearchResult]:
    trimmed = []
    for position, result in enumerate(results[:top_k], start=1):
        result.rank = position
        trimmed.append(result)
    return trimmed


def hybrid_search(expanded: ExpandedQuery, vector_fn, fts_fn, **kwargs) -> HybridSearchOutcome:
    """Synchronous wrapper around :func:`hybrid_search_async`."""
    return run_async(hybrid_search_async(expanded, vector_fn, fts_fn, **kwargs))


def run_async(coro):
    """Run a coroutine from sync code, even inside a running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # Already inside an event loop (e.g. an async MCP host): use a private one
    # in a worker thread so we never block or reuse the caller's loop.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
