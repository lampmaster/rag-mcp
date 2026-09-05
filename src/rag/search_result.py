"""Unified internal representation shared by every retriever.

Both the FAISS vector retriever and the SQLite FTS5 retriever convert their
raw output into `SearchResult`, so the fusion layer never depends on FAISS or
sqlite3 structures directly.
"""

from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """A single retrieved chunk.

    rank      1-based position inside the retriever that produced it
    score     retriever-native score (cosine similarity / BM25), for logging.
              Scales differ between retrievers - never combine them directly.
    retriever name of the retriever ("vector" / "fts"), for logging
    """

    chunk_id: str
    text: str
    source: str
    rank: int
    score: float | None = None
    retriever: str = ""
    metadata: dict = field(default_factory=dict)

    def to_chunk(self) -> dict:
        """Convert back to the chunk dict shape used by the existing pipeline."""
        chunk = {
            "text": self.text,
            "source": self.source,
            "chunk_id": self.chunk_id,
        }
        chunk.update(self.metadata)
        return chunk
