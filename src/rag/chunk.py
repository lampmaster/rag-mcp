import tiktoken
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHUNK_SIZE, CHUNK_OVERLAP

encoder = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str):
    """Split text into chunks with overlap."""
    tokens = encoder.encode(text)
    chunks = []

    step = CHUNK_SIZE - CHUNK_OVERLAP
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i + CHUNK_SIZE]
        chunks.append(encoder.decode(chunk_tokens))

    return chunks


def make_chunk_id(source: str, index: int) -> str:
    """Build a stable chunk id shared by the FAISS metadata and the FTS5 index.

    The id only depends on the document path and the position of the chunk
    inside that document, so rebuilding the index over unchanged documents
    produces exactly the same ids in both stores.
    """
    return f"{source}::{index}"


def chunk_documents(documents):
    """Chunk all documents into smaller pieces."""
    all_chunks = []

    for doc in documents:
        chunks = chunk_text(doc["text"])
        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "source": doc["path"],
                # Stable, globally unique id (used by FAISS metadata, FTS5 and RRF)
                "chunk_id": make_chunk_id(doc["path"], idx),
                # Position of the chunk inside its own document
                "chunk_index": idx,
            })

    return all_chunks


if __name__ == "__main__":
    # Test chunking
    test_doc = {"text": "This is a test document. " * 100, "path": "test.txt"}
    chunks = chunk_documents([test_doc])
    print(f"Created {len(chunks)} chunks")
