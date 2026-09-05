"""High-level retrieval orchestration and answer generation.

Runtime pipeline:

    user query
        ↓
    Query Expansion (rag/query_expansion.py, separate Qwen call)
        ↓
          ┌── FAISS vector search (original query) ──┐
          │                                          ├── RRF ── Top-K chunks
          └── SQLite FTS5 (query + keywords) ────────┘
        ↓
    build_prompt() → ask_llm()   (unchanged answer generation)
"""

import logging
import sys
import pickle
from pathlib import Path

import faiss
import requests
from sentence_transformers import SentenceTransformer

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CHUNKS_PATH,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    FINAL_TOP_K,
    FTS_CANDIDATES,
    HYBRID_SEARCH_ENABLED,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    OLLAMA_URL,
    QUERY_EXPANSION_ENABLED,
    RRF_K,
    TOP_K,
    VECTOR_CANDIDATES,
)
from rag.fts import default_db_path as fts_db_path
from rag.fts import fts_search as _fts_search
from rag.hybrid_search import (
    AllRetrieversFailedError,
    HybridSearchOutcome,
    hybrid_search,
)
from rag.query_expansion import ExpandedQuery, expand_query, fallback_expansion
from rag.search_result import SearchResult

logger = logging.getLogger(__name__)

model = SentenceTransformer(EMBEDDING_MODEL)

# Global variables for index and chunks
index = None
chunks = []


def _chunks_are_current(loaded) -> bool:
    """Detect indexes built before stable string chunk_ids were introduced."""
    if not loaded:
        return False
    first = loaded[0]
    return isinstance(first, dict) and isinstance(first.get("chunk_id"), str)


def _ensure_index_exists():
    """Ensure FAISS + FTS indexes exist, build them if they don't."""
    global index, chunks

    # Resolve paths relative to src directory
    src_dir = Path(__file__).parent.parent
    index_path = src_dir / FAISS_INDEX_PATH
    chunks_path = src_dir / CHUNKS_PATH

    # Check if index exists
    if index_path.exists() and chunks_path.exists():
        try:
            loaded_index = faiss.read_index(str(index_path))
            with open(chunks_path, "rb") as f:
                loaded_chunks = pickle.load(f)
            if _chunks_are_current(loaded_chunks) and fts_db_path().exists():
                index = loaded_index
                chunks = loaded_chunks
                return True
            print("📦 Index is outdated (missing stable chunk ids or FTS index).")
            print("Rebuilding index...")
        except Exception as e:
            print(f"⚠️  Warning: Error loading existing index: {e}")
            print("Rebuilding index...")
    else:
        print("📦 Index not found. Building index from documents...")

    try:
        from rag.build_index import build_index
        build_index()

        # Load the newly created index
        if index_path.exists() and chunks_path.exists():
            index = faiss.read_index(str(index_path))
            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)
            print("✅ Index built and loaded successfully")
            return True
        else:
            print("❌ Failed to build index. No documents found or error occurred.")
            from config import DOCUMENTS_DIR
            docs_path = src_dir / DOCUMENTS_DIR
            print(f"   Check that documents exist in: {docs_path}")
            return False
    except Exception as e:
        print(f"❌ Error building index: {e}")
        import traceback
        traceback.print_exc()
        return False


# Initialize index on module load
_ensure_index_exists()


# --------------------------------------------------------------------------
# Retrievers (unified SearchResult output)
# --------------------------------------------------------------------------

def vector_search(query: str, limit: int = None) -> list[SearchResult]:
    """FAISS semantic retrieval. Always embeds the ORIGINAL user query."""
    limit = VECTOR_CANDIDATES if limit is None else limit

    if index is None or len(chunks) == 0:
        if not _ensure_index_exists():
            return []
    if index is None or len(chunks) == 0:
        return []

    q_emb = model.encode([query])
    faiss.normalize_L2(q_emb)

    scores, ids = index.search(q_emb, min(limit, len(chunks)))

    results = []
    rank = 0
    for chunk_pos, score in zip(ids[0], scores[0]):
        if chunk_pos < 0 or chunk_pos >= len(chunks):
            continue
        chunk = chunks[chunk_pos]
        rank += 1
        results.append(
            SearchResult(
                chunk_id=str(chunk["chunk_id"]),
                text=chunk["text"],
                source=chunk["source"],
                rank=rank,
                score=float(score),
                retriever="vector",
            )
        )
    return results


def fts_search(query: str, limit: int = None) -> list[SearchResult]:
    """SQLite FTS5 lexical retrieval (query text = original query + keywords)."""
    return _fts_search(query, FTS_CANDIDATES if limit is None else limit)


# --------------------------------------------------------------------------
# Retrieval entry points
# --------------------------------------------------------------------------

def retrieve_hybrid(query: str, top_k: int = None, expand: bool = None) -> HybridSearchOutcome:
    """Query expansion + parallel vector/FTS retrieval + RRF fusion."""
    top_k = FINAL_TOP_K if top_k is None else top_k
    expand = QUERY_EXPANSION_ENABLED if expand is None else expand

    expanded = expand_query(query) if expand else fallback_expansion(query)
    if expanded.is_fallback:
        logger.info("Retrieving with the original query only (no expansion terms)")

    return hybrid_search(
        expanded,
        vector_search,
        fts_search,
        vector_candidates=VECTOR_CANDIDATES,
        fts_candidates=FTS_CANDIDATES,
        top_k=top_k,
        rrf_k=RRF_K,
    )


def retrieve_vector_only(query: str, top_k: int = None) -> list[dict]:
    """Original vector-only retrieval (kept for comparison/benchmarking)."""
    top_k = TOP_K if top_k is None else top_k
    return [r.to_chunk() for r in vector_search(query, top_k)]


def retrieve(query: str, top_k: int = None, hybrid: bool = None) -> list[dict]:
    """Retrieve relevant chunks for a query.

    Returns the same chunk dicts as before (`text`, `source`, `chunk_id`), so
    the existing prompt builder, CLI and MCP layers are unaffected.
    """
    hybrid = HYBRID_SEARCH_ENABLED if hybrid is None else hybrid
    if not hybrid:
        return retrieve_vector_only(query, top_k)

    try:
        outcome = retrieve_hybrid(query, top_k=top_k)
    except AllRetrieversFailedError as exc:
        logger.error("Hybrid retrieval failed completely: %s", exc)
        return []
    except Exception as exc:  # never break the CLI/MCP on a retrieval problem
        logger.error("Hybrid retrieval error (%s) - falling back to vector only", exc)
        try:
            return retrieve_vector_only(query, top_k)
        except Exception:
            return []

    return [r.to_chunk() for r in outcome.results]


def build_prompt(query, contexts):
    """Build prompt with retrieved context."""
    if not contexts:
        return f"""
<role>You are a helpful assistant that answers questions about company information.</role>
<instructions>Answer the question based on your general knowledge. If you don't know, say so.</instructions>

<query>
{query}
</query>

<assistant>
"""

    context_text = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}"
        for c in contexts
    )

    return f"""
<role>You are a helpful assistant that answers questions about company information.</role>
<instructions>Answer the question ONLY based on the context provided below. If the answer is not in the context, say "I don't have that information in the knowledge base."</instructions>

<context>
{context_text}
</context>

<query>
{query}
</query>

<assistant>
"""


def ask_llm(prompt):
    """Query Ollama LLM."""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=OLLAMA_TIMEOUT,
    )
    return response.json()["response"]


def ask(query: str):
    """Answer a question using RAG."""
    contexts = retrieve(query)
    prompt = build_prompt(query, contexts)
    return ask_llm(prompt), contexts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    while True:
        q = input("\n❓ Question: ")
        if q.lower() in {"exit", "quit"}:
            break
        print("\n🤖 Answer:\n")
        answer, sources = ask(q)
        print(answer)
        if sources:
            print("\n📚 Sources:")
            for src in sources:
                print(f"  - {src['source']}")
