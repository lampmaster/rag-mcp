"""SQLite FTS5 lexical retrieval.

Second, independent retriever next to FAISS. It is built from exactly the same
chunks as the vector index (no second chunking pass) and keyed by the same
stable `chunk_id`, which is what makes RRF fusion and dedup possible.
"""

import logging
import re
import sqlite3
import sys
import time
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FTS_INDEX_PATH, FTS_MAX_TERMS, FTS_TOKENIZE
from rag.search_result import SearchResult

logger = logging.getLogger(__name__)

FTS_TABLE = "chunks_fts"

# Column weights for bm25(): chunk_id (unindexed), source, text.
# The source path is weighted higher so that "where is docker.md" ranks the
# docker.md document itself first.
BM25_WEIGHTS = (0.0, 2.0, 1.0)

# Token = a word possibly containing '.', '-', '_' or '/' (docker.md,
# AUTH_TOKEN, docker-compose, docs/docker.md).
_TERM_RE = re.compile(r"[^\W_][\w./\-]*", re.UNICODE)


class FTSUnavailableError(Exception):
    """Raised when the FTS index is missing or unreadable."""


def default_db_path() -> Path:
    """Resolve the FTS database path (relative paths resolve to src/)."""
    return Path(__file__).parent.parent / FTS_INDEX_PATH


# --------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------

def build_fts_index(chunks, db_path=None) -> int:
    """(Re)build the FTS5 index from the *same* chunks used for FAISS.

    Returns the number of indexed chunks.
    """
    path = Path(db_path) if db_path else default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    try:
        conn.execute(f"DROP TABLE IF EXISTS {FTS_TABLE}")
        conn.execute(
            f"CREATE VIRTUAL TABLE {FTS_TABLE} USING fts5("
            f"chunk_id UNINDEXED, source, text, tokenize=\"{FTS_TOKENIZE}\")"
        )
        conn.executemany(
            f"INSERT INTO {FTS_TABLE} (chunk_id, source, text) VALUES (?, ?, ?)",
            [
                (str(c["chunk_id"]), str(c.get("source", "")), c.get("text", ""))
                for c in chunks
            ],
        )
        conn.commit()
    finally:
        conn.close()

    logger.info("FTS5 index built: %s chunks -> %s", len(chunks), path)
    return len(chunks)


def fts_index_exists(db_path=None) -> bool:
    path = Path(db_path) if db_path else default_db_path()
    if not path.exists():
        return False
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE name = ?", (FTS_TABLE,)
            ).fetchone()
            return row is not None
        finally:
            conn.close()
    except sqlite3.Error:
        return False


# --------------------------------------------------------------------------
# Query building
# --------------------------------------------------------------------------

def tokenize_query(text: str) -> list[str]:
    """Split free text into FTS-usable terms."""
    terms = []
    for match in _TERM_RE.findall(text or ""):
        term = match.strip("._-/")
        if len(term) >= 2:
            terms.append(term)
    return terms


def build_fts_query(text: str, keywords=None, max_terms: int = None) -> str:
    """Build an FTS5 MATCH expression from the original query + keywords.

    Terms are quoted (so punctuation cannot break the FTS5 grammar) and joined
    with OR; each is a prefix phrase so that a token followed by punctuation in
    the document ("see docker.md.") still matches.
    """
    max_terms = max_terms or FTS_MAX_TERMS

    terms: list[str] = []
    seen: set[str] = set()
    # Keywords first: they are the high-signal terms and survive the cap.
    for source_text in list(keywords or []) + [text or ""]:
        for term in tokenize_query(source_text):
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            terms.append(term)
            if len(terms) >= max_terms:
                break
        if len(terms) >= max_terms:
            break

    if not terms:
        return ""

    return " OR ".join(f'"{t.replace(chr(34), chr(34) * 2)}" *' for t in terms)


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

def fts_search(query: str, limit: int, db_path=None) -> list[SearchResult]:
    """Lexical retrieval over the FTS5 index.

    `query` is free text (original user query + expansion keywords); it is
    converted into a MATCH expression internally.
    Raises FTSUnavailableError when the index cannot be queried.
    """
    match_expr = query if _looks_like_match_expr(query) else build_fts_query(query)
    if not match_expr:
        logger.info("FTS search skipped: empty query")
        return []

    path = Path(db_path) if db_path else default_db_path()
    if not path.exists():
        raise FTSUnavailableError(f"FTS index not found: {path}")

    started = time.perf_counter()
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise FTSUnavailableError(f"cannot open FTS index {path}: {exc}") from exc

    try:
        weights = ", ".join(str(w) for w in BM25_WEIGHTS)
        rows = conn.execute(
            f"SELECT chunk_id, source, text, bm25({FTS_TABLE}, {weights}) AS score "
            f"FROM {FTS_TABLE} WHERE {FTS_TABLE} MATCH ? "
            f"ORDER BY score LIMIT ?",
            (match_expr, int(limit)),
        ).fetchall()
    except sqlite3.Error as exc:
        raise FTSUnavailableError(f"FTS query failed: {exc}") from exc
    finally:
        conn.close()

    results = [
        SearchResult(
            chunk_id=row[0],
            source=row[1],
            text=row[2],
            rank=i,
            score=row[3],
            retriever="fts",
        )
        for i, row in enumerate(rows, start=1)
    ]
    logger.info(
        "FTS search: %s candidates in %.1f ms",
        len(results), (time.perf_counter() - started) * 1000,
    )
    return results


def _looks_like_match_expr(query: str) -> bool:
    """True when the caller already passed a prepared MATCH expression."""
    return bool(query) and '" *' in query
