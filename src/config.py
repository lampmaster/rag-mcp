# Configuration for Company Knowledge Base Assistant
#
# Every value can be overridden with an environment variable of the same name,
# which keeps the defaults below working exactly as before while letting tools
# (e.g. the benchmark runner) point the pipeline at a different corpus/index.

import os


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Document directory - update this to point to your company documentation
DOCUMENTS_DIR = _env_str("DOCUMENTS_DIR", "./docs")

# Chunking configuration
CHUNK_SIZE = _env_int("CHUNK_SIZE", 700)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 100)

# Embedding model
EMBEDDING_MODEL = _env_str("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# FAISS index paths (relative to src directory)
FAISS_INDEX_PATH = _env_str("FAISS_INDEX_PATH", "index.faiss")
CHUNKS_PATH = _env_str("CHUNKS_PATH", "chunks.pkl")

# SQLite FTS5 full-text index path (relative to src directory)
FTS_INDEX_PATH = _env_str("FTS_INDEX_PATH", "fts_index.db")

# FTS5 tokenizer.
# `tokenchars` keeps '_', '.' and '-' inside tokens so that exact technical
# terms survive tokenization as single tokens:
#   docker.md, AUTH_TOKEN, docker-compose, K8S_NAMESPACE, DB_POOL_SIZE
# '/' is deliberately NOT a token char so that a source path such as
# "docs/docker.md" still yields a "docker.md" token.
FTS_TOKENIZE = _env_str("FTS_TOKENIZE", "unicode61 remove_diacritics 2 tokenchars '_.-'")

# Ollama configuration
OLLAMA_URL = _env_str("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = _env_str("OLLAMA_MODEL", "qwen3:0.6b")

# Ollama request timeouts (seconds)
OLLAMA_TIMEOUT = _env_float("OLLAMA_TIMEOUT", 120.0)
QUERY_EXPANSION_TIMEOUT = _env_float("QUERY_EXPANSION_TIMEOUT", 60.0)

# Query Expansion configuration
QUERY_EXPANSION_ENABLED = _env_bool("QUERY_EXPANSION_ENABLED", True)
# Deterministic generation - small Qwen models hallucinate/malform JSON otherwise.
QUERY_EXPANSION_TEMPERATURE = _env_float("QUERY_EXPANSION_TEMPERATURE", 0.0)
# Maximum number of LLM calls for query expansion (1 initial + 1 retry).
QUERY_EXPANSION_MAX_ATTEMPTS = _env_int("QUERY_EXPANSION_MAX_ATTEMPTS", 2)
MAX_KEYWORDS = _env_int("MAX_KEYWORDS", 5)
MAX_ALTERNATIVE_QUERIES = _env_int("MAX_ALTERNATIVE_QUERIES", 2)
# How much of the previous (invalid) model output is fed back into the retry prompt.
PREVIOUS_OUTPUT_LIMIT = _env_int("PREVIOUS_OUTPUT_LIMIT", 800)
# Qwen3 emits <think> blocks; ask Ollama to disable them for the expansion call.
OLLAMA_DISABLE_THINKING = _env_bool("OLLAMA_DISABLE_THINKING", True)

# RAG retrieval configuration
TOP_K = _env_int("TOP_K", 5)

# Hybrid retrieval configuration
HYBRID_SEARCH_ENABLED = _env_bool("HYBRID_SEARCH_ENABLED", True)
VECTOR_CANDIDATES = _env_int("VECTOR_CANDIDATES", 10)
FTS_CANDIDATES = _env_int("FTS_CANDIDATES", 10)
FINAL_TOP_K = _env_int("FINAL_TOP_K", TOP_K)
RRF_K = _env_int("RRF_K", 60)
# Maximum number of terms sent to SQLite FTS5 in a single MATCH expression.
FTS_MAX_TERMS = _env_int("FTS_MAX_TERMS", 32)

# Logging - set RAG_LOG_LEVEL=INFO to see expansion keywords, retriever
# latencies, candidate counts, fused chunk ids and fallback events.
RAG_LOG_LEVEL = _env_str("RAG_LOG_LEVEL", "WARNING")
