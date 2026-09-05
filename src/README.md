# Company Knowledge Base Assistant

An intelligent Q&A system that answers questions about company documentation using RAG (Retrieval-Augmented Generation) and MCP (Model Context Protocol) tools.

## Features

- **Hybrid retrieval**: FAISS semantic search + SQLite FTS5 full-text search, merged with Reciprocal Rank Fusion
- **Query Expansion**: a separate local Qwen call turns a noisy question into keywords and alternative queries
- **MCP tools**: Dynamic document reading and management
- **Local LLM**: Privacy-preserving answers using Ollama

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Documents

Create a `docs/` directory and add your company documentation files (`.txt`, `.md`, `.pdf`, `.docx`):

```bash
mkdir docs
# Add your company documentation files here
```

### 3. Configure

Edit `config.py` to set:
- `DOCUMENTS_DIR`: Path to your documentation directory
- `OLLAMA_MODEL`: Local LLM model to use (default: `qwen3:0.6b`)
- `VECTOR_CANDIDATES` / `FTS_CANDIDATES`: candidates per retriever (default 10)
- `FINAL_TOP_K`: chunks handed to the answering LLM after fusion (default 5)
- `RRF_K`: Reciprocal Rank Fusion constant (default 60)
- `QUERY_EXPANSION_ENABLED`, `HYBRID_SEARCH_ENABLED`: feature switches
- `RAG_LOG_LEVEL`: set to `INFO` to log keywords, latencies and fused chunk ids

Every value in `config.py` can also be overridden with an environment variable
of the same name.

### 4. Build Index (Optional)

The index will be built automatically on first use. To manually build it:

```bash
python main.py build-index
```

Or directly:

```bash
python -m rag.build_index
```

## Usage

### Interactive CLI

Run the interactive assistant:

```bash
python main.py
```

Then ask questions about your company documentation!

## Project Structure

```
src/
├── config.py              # Configuration
├── main.py                # CLI entry point
├── assistant.py           # Main assistant class
├── rag/                   # RAG components
│   ├── ingest.py         # Document ingestion
│   ├── chunk.py          # Text chunking + stable chunk_id
│   ├── embed.py          # Embedding generation
│   ├── build_index.py    # FAISS + FTS5 index building
│   ├── query_expansion.py # LLM query expansion, parsing, validation, retry, fallback
│   ├── fts.py            # SQLite FTS5 indexing and lexical retrieval
│   ├── hybrid_search.py  # Parallel retrieval, RRF, graceful degradation
│   ├── search_result.py  # Unified SearchResult used by both retrievers
│   └── query.py          # Orchestration, prompt building, answer generation
├── mcp/                   # MCP components
│   ├── server.py         # MCP server with tools
│   └── client.py         # MCP client
├── tests/                 # Unit tests (pytest)
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## How It Works

### Indexing

```
Documents
    ↓
Existing chunking (one pass, stable chunk_id per chunk)
    ↓
Chunks
    ├────────────→ Embeddings → FAISS      (index.faiss + chunks.pkl)
    │
    └────────────→ Text       → SQLite FTS5 (fts_index.db)
```

Both indexes are built from exactly the same chunks and keyed by the same
stable `chunk_id` (`<source>::<chunk index>`), which is what makes fusion and
deduplication possible.

### Retrieval

```
User question
    ↓
Query Expansion  (separate Qwen call, temperature 0.0, max 2 attempts)
    ↓  keywords + alternative queries (or a safe empty fallback)
    ↓
          ┌── FAISS vector search   (original query)             ──┐
          ┤                                                        ├── RRF ── Top-5
          └── SQLite FTS5 search    (original query + keywords)  ──┘
    ↓
existing context builder → existing Ollama answer generation (+ MCP tools)
```

The two retrievers run concurrently (`asyncio.gather` over `asyncio.to_thread`),
so full-text latency is not added on top of vector latency. If one retriever
fails, the results of the other are used unchanged; if query expansion fails,
retrieval continues with the original query.

## Running the tests

```bash
python -m pytest tests -q
```

## Benchmark

See [`../benchmark/README.md`](../benchmark/README.md) for a reproducible
vector-only vs hybrid comparison.

## MCP Tools

The MCP server provides:
- `read_document`: Read a specific document
- `list_documents`: List all available documents
- `search_documents`: Search documents by name

## Troubleshooting

**Index not found**: Run `python main.py build-index` first

**Ollama not responding**: Make sure Ollama is running and the model is installed:
```bash
ollama pull llama3
```

**No documents found**: Check that `DOCUMENTS_DIR` in `config.py` points to your documents

## License

MIT
