# Hybrid Retrieval Benchmark

Reproducible comparison of the **existing vector-only FAISS retrieval** against
the **new hybrid retrieval** (Query Expansion + FAISS + SQLite FTS5 + RRF).

```
benchmark/
├── docs/                   5 realistic technical documents
│   ├── docker.md           docker-compose, DOCKER_HOST, docker compose up, Dockerfile
│   ├── security.md         AUTH_TOKEN, JWT_SECRET, Bearer
│   ├── database.md         DB_POOL_SIZE, DATABASE_URL, PostgreSQL
│   ├── kubernetes.md       K8S_NAMESPACE, kubectl apply, ConfigMap
│   └── mcp.md              MCP, Model Context Protocol, MCP server
├── benchmark_queries.json  18 queries + ground truth (evaluator only)
└── run_benchmark.py        the runner
```

## Running

```bash
# from the repository root
src/venv/bin/python benchmark/run_benchmark.py --model qwen3:1.7b --rebuild
```

The runner builds its own index under `benchmark/.index/` (it never touches
`src/index.faiss` or `src/fts_index.db`) by overriding `DOCUMENTS_DIR`,
`FAISS_INDEX_PATH`, `CHUNKS_PATH` and `FTS_INDEX_PATH` through the environment.

Useful flags:

| flag | meaning |
| --- | --- |
| `--model` | Ollama model used for Query Expansion (default: `config.OLLAMA_MODEL`) |
| `--no-expansion` | run hybrid retrieval on the raw query, to isolate the effect of expansion |
| `--rebuild` | rebuild the benchmark index from scratch |
| `--top-k` | K for Recall@K (default 5) |
| `--rank-depth` | how deep to look for the relevant chunk when reporting ranks (default 20) |
| `--chunk-size` / `--chunk-overlap` | benchmark chunking (default 250/50, smaller than the app default so that 5 documents produce a corpus where top-5 is discriminative) |

## Ground truth

Each entry declares the document and the exact term that must be retrieved:

```json
{
  "query": "Hi! My name is Alex! Where is AUTH_TOKEN?",
  "expected_document": "security.md",
  "expected_term": "AUTH_TOKEN",
  "category": "noisy"
}
```

A chunk counts as relevant when it comes from `expected_document` **and**
contains `expected_term` (case-insensitive); `expected_term: null` means any
chunk of that document counts. The ground truth is read only by
`run_benchmark.py` - nothing in `src/rag/` knows about it.

Categories: `filename`, `env variable`, `abbreviation`, `command`, `semantic`
(the target term does not appear in the question), `noisy` (conversational
greeting + personal information around a technical term).

## Reported metrics

* Recall@1 / Recall@3 / Recall@5 for both pipelines
* rank of the first relevant chunk per query (`-` when not found within `--rank-depth`)
* MRR
* latency: vector-only, hybrid retrieval, and query expansion separately
* per-category breakdown
* an explicit list of queries where hybrid performed *worse* than vector-only

Raw per-query results (including the generated keywords) are written to
`benchmark/results.json`.
