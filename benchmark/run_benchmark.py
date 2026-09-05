#!/usr/bin/env python3
"""Reproducible benchmark: vector-only FAISS retrieval vs hybrid retrieval.

    python benchmark/run_benchmark.py --model qwen3:1.7b

The runner builds a dedicated index over `benchmark/docs` (it never touches the
`src/index.faiss` / `src/fts_index.db` of the real knowledge base), then runs
every query from `benchmark_queries.json` through both pipelines and prints a
comparison table plus summary metrics.

Ground truth lives in benchmark_queries.json and is used only here.
"""

import argparse
import json
import logging
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
SRC_DIR = BENCHMARK_DIR.parent / "src"
INDEX_DIR = BENCHMARK_DIR / ".index"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Ollama model for query expansion (default: config.OLLAMA_MODEL)")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the benchmark index from scratch")
    parser.add_argument("--no-expansion", action="store_true", help="run hybrid retrieval without query expansion")
    parser.add_argument("--rank-depth", type=int, default=20, help="how deep to look for the relevant chunk (default: 20)")
    parser.add_argument("--top-k", type=int, default=5, help="Recall@K (default: 5)")
    parser.add_argument("--candidates", type=int, default=10, help="candidates per retriever (default: 10)")
    parser.add_argument("--chunk-size", type=int, default=250, help="benchmark chunk size in tokens (default: 250)")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="benchmark chunk overlap (default: 50)")
    parser.add_argument("--output", default=str(BENCHMARK_DIR / "results.json"), help="where to write the raw results")
    parser.add_argument("--verbose", action="store_true", help="show retrieval logs")
    return parser.parse_args()


def configure_environment(args):
    """Point the pipeline at the benchmark corpus *before* importing config."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["DOCUMENTS_DIR"] = str(BENCHMARK_DIR / "docs")
    os.environ["FAISS_INDEX_PATH"] = str(INDEX_DIR / "index.faiss")
    os.environ["CHUNKS_PATH"] = str(INDEX_DIR / "chunks.pkl")
    os.environ["FTS_INDEX_PATH"] = str(INDEX_DIR / "fts_index.db")
    # Smaller chunks than the default 700 so that the 5 documents produce a
    # corpus where top-5 retrieval is actually discriminative. Both pipelines
    # use exactly the same chunks.
    os.environ["CHUNK_SIZE"] = str(args.chunk_size)
    os.environ["CHUNK_OVERLAP"] = str(args.chunk_overlap)
    os.environ["VECTOR_CANDIDATES"] = str(args.candidates)
    os.environ["FTS_CANDIDATES"] = str(args.candidates)
    os.environ["FINAL_TOP_K"] = str(args.top_k)
    if args.model:
        os.environ["OLLAMA_MODEL"] = args.model
    if args.rebuild and INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(SRC_DIR))


# --------------------------------------------------------------------------
# Evaluation helpers (ground truth lives here, never in the retrieval code)
# --------------------------------------------------------------------------

def relevant_chunk_ids(chunks, entry):
    """Chunk ids that count as a correct answer for one benchmark query."""
    document = entry["expected_document"]
    term = (entry.get("expected_term") or "").lower()
    relevant = set()
    for chunk in chunks:
        if Path(chunk["source"]).name != document:
            continue
        if term and term not in chunk["text"].lower():
            continue
        relevant.add(chunk["chunk_id"])
    return relevant


def first_hit_rank(results, relevant):
    for position, result in enumerate(results, start=1):
        if result.chunk_id in relevant:
            return position
    return None


def fmt_rank(rank):
    return str(rank) if rank else "-"


def mean(values):
    return statistics.fmean(values) if values else float("nan")


# --------------------------------------------------------------------------

def main():
    args = parse_args()
    configure_environment(args)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    import config
    from rag.hybrid_search import hybrid_search
    from rag.query import fts_search, vector_search
    from rag.query_expansion import expand_query, fallback_expansion
    import pickle

    with open(config.CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    entries = json.loads((BENCHMARK_DIR / "benchmark_queries.json").read_text())["queries"]

    print("=" * 100)
    print("Hybrid Retrieval Benchmark")
    print("=" * 100)
    print(f"Documents        : {len(set(c['source'] for c in chunks))} files in {config.DOCUMENTS_DIR}")
    print(f"Chunks indexed   : {len(chunks)} (chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")
    print(f"Queries          : {len(entries)}")
    print(f"Candidates       : vector={config.VECTOR_CANDIDATES}, fts={config.FTS_CANDIDATES}, RRF k={config.RRF_K}")
    print(f"Final top-k      : {args.top_k} (ranks measured down to depth {args.rank_depth})")
    print(f"Expansion model  : {config.OLLAMA_MODEL if not args.no_expansion else 'disabled'}")
    print()

    # Warm up the embedding model so the first query is not penalised.
    vector_search("warm up", 1)

    rows = []
    for entry in entries:
        query = entry["query"]
        relevant = relevant_chunk_ids(chunks, entry)
        if not relevant:
            print(f"⚠️  No chunk matches the ground truth for {query!r} - skipped")
            continue

        # --- 1. existing vector-only retrieval -----------------------------
        started = time.perf_counter()
        vector_results = vector_search(query, args.rank_depth)
        vector_ms = (time.perf_counter() - started) * 1000

        # --- 2. new hybrid retrieval ---------------------------------------
        started = time.perf_counter()
        expanded = fallback_expansion(query) if args.no_expansion else expand_query(query)
        expansion_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        outcome = hybrid_search(
            expanded,
            vector_search,
            fts_search,
            vector_candidates=config.VECTOR_CANDIDATES,
            fts_candidates=config.FTS_CANDIDATES,
            top_k=args.rank_depth,
        )
        retrieval_ms = (time.perf_counter() - started) * 1000

        rows.append({
            "query": query,
            "category": entry["category"],
            "expected_document": entry["expected_document"],
            "expected_term": entry.get("expected_term"),
            "keywords": expanded.keywords,
            "alternatives": expanded.alternatives,
            "expansion_fallback": expanded.is_fallback,
            "vector_rank": first_hit_rank(vector_results, relevant),
            "hybrid_rank": first_hit_rank(outcome.results, relevant),
            "vector_hit_at_k": bool(first_hit_rank(vector_results[:args.top_k], relevant)),
            "hybrid_hit_at_k": bool(first_hit_rank(outcome.results[:args.top_k], relevant)),
            "vector_latency_ms": vector_ms,
            "expansion_latency_ms": expansion_ms,
            "hybrid_retrieval_latency_ms": retrieval_ms,
            "hybrid_total_latency_ms": expansion_ms + retrieval_ms,
            "fusion_used": outcome.fusion_used,
            "fts_candidates": len(outcome.fts_results),
            "vector_candidates": len(outcome.vector_results),
            "top_chunks": [r.chunk_id for r in outcome.results[:args.top_k]],
        })

    print_report(rows, args)

    Path(args.output).write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"\nRaw results written to {args.output}")


def print_report(rows, args):
    k = args.top_k
    width_q = 46
    header = (
        f"{'Query':<{width_q}} {'Category':<13} {'Vector Rank':>11} {'Hybrid Rank':>11} "
        f"{'V ms':>7} {'H ms':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        query = row["query"]
        if len(query) > width_q - 1:
            query = query[: width_q - 2] + "…"
        print(
            f"{query:<{width_q}} {row['category']:<13} "
            f"{fmt_rank(row['vector_rank']):>11} {fmt_rank(row['hybrid_rank']):>11} "
            f"{row['vector_latency_ms']:>7.0f} {row['hybrid_total_latency_ms']:>8.0f}"
        )

    total = len(rows)
    v_hits = sum(r["vector_hit_at_k"] for r in rows)
    h_hits = sum(r["hybrid_hit_at_k"] for r in rows)

    def recall_at(field, depth):
        return sum(1 for r in rows if r[field] and r[field] <= depth) / total * 100
    v_ranks = [r["vector_rank"] for r in rows if r["vector_rank"]]
    h_ranks = [r["hybrid_rank"] for r in rows if r["hybrid_rank"]]
    v_mrr = mean([1 / r["vector_rank"] if r["vector_rank"] else 0.0 for r in rows])
    h_mrr = mean([1 / r["hybrid_rank"] if r["hybrid_rank"] else 0.0 for r in rows])

    print()
    print("=" * 100)
    print(f"Vector-only Recall@{k}: {v_hits / total * 100:.1f}%  ({v_hits}/{total})")
    print(f"Hybrid Recall@{k}:      {h_hits / total * 100:.1f}%  ({h_hits}/{total})")
    print()
    print(f"{'':<14}{'Recall@1':>10}{'Recall@3':>10}{'Recall@5':>10}")
    print(f"{'Vector-only':<14}{recall_at('vector_rank', 1):>9.1f}%{recall_at('vector_rank', 3):>9.1f}%"
          f"{recall_at('vector_rank', 5):>9.1f}%")
    print(f"{'Hybrid':<14}{recall_at('hybrid_rank', 1):>9.1f}%{recall_at('hybrid_rank', 3):>9.1f}%"
          f"{recall_at('hybrid_rank', 5):>9.1f}%")
    print()
    print(f"Average Vector Rank:  {mean(v_ranks):.2f}  (found in {len(v_ranks)}/{total} within depth {args.rank_depth})")
    print(f"Average Hybrid Rank:  {mean(h_ranks):.2f}  (found in {len(h_ranks)}/{total} within depth {args.rank_depth})")
    print(f"Vector MRR:           {v_mrr:.3f}")
    print(f"Hybrid MRR:           {h_mrr:.3f}")
    print()
    print(f"Average Vector-only latency: {mean([r['vector_latency_ms'] for r in rows]):.0f} ms")
    print(f"Average Hybrid latency:      {mean([r['hybrid_total_latency_ms'] for r in rows]):.0f} ms"
          f"  (retrieval {mean([r['hybrid_retrieval_latency_ms'] for r in rows]):.0f} ms"
          f" + query expansion {mean([r['expansion_latency_ms'] for r in rows]):.0f} ms)")

    print()
    print("Per category")
    print("-" * 100)
    print(f"{'Category':<15} {'N':>3} {'Vector Recall@' + str(k):>17} {'Hybrid Recall@' + str(k):>17} "
          f"{'Avg V rank':>11} {'Avg H rank':>11}")
    categories = sorted({r["category"] for r in rows})
    for category in categories:
        subset = [r for r in rows if r["category"] == category]
        vr = [r["vector_rank"] for r in subset if r["vector_rank"]]
        hr = [r["hybrid_rank"] for r in subset if r["hybrid_rank"]]
        print(
            f"{category:<15} {len(subset):>3} "
            f"{sum(r['vector_hit_at_k'] for r in subset) / len(subset) * 100:>16.0f}% "
            f"{sum(r['hybrid_hit_at_k'] for r in subset) / len(subset) * 100:>16.0f}% "
            f"{mean(vr):>11.1f} {mean(hr):>11.1f}"
        )

    regressions = [r for r in rows if r["vector_hit_at_k"] and not r["hybrid_hit_at_k"]]
    worse = [
        r for r in rows
        if r["vector_rank"] and r["hybrid_rank"] and r["hybrid_rank"] > r["vector_rank"]
    ]
    print()
    print(f"Queries where hybrid lost the hit inside top-{k}: {len(regressions)}")
    for row in regressions:
        print(f"  - [{row['category']}] {row['query']}")
    print(f"Queries where hybrid ranks the relevant chunk lower than vector-only: {len(worse)}")
    for row in worse:
        print(f"  - [{row['category']}] {row['query']} (vector {row['vector_rank']} -> hybrid {row['hybrid_rank']})")

    fallbacks = [r for r in rows if r["expansion_fallback"]]
    print(f"\nQuery expansion fallbacks: {len(fallbacks)}/{total}")
    for row in fallbacks:
        print(f"  - {row['query']}")


if __name__ == "__main__":
    main()
