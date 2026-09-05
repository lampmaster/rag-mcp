import faiss
import pickle
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.ingest import ingest_documents
from rag.chunk import chunk_documents
from rag.embed import embed_chunks
from rag.fts import build_fts_index, default_db_path
from config import FAISS_INDEX_PATH, CHUNKS_PATH


def build_index():
    """Build the FAISS vector index and the SQLite FTS5 index from documents.

    Documents are chunked exactly once; the same chunks (and the same stable
    chunk_ids) feed both the embeddings/FAISS side and the FTS5 side.
    """
    # Resolve paths relative to src directory
    src_dir = Path(__file__).parent.parent
    index_path = src_dir / FAISS_INDEX_PATH
    chunks_path = src_dir / CHUNKS_PATH
    fts_path = default_db_path()

    print("📥 Loading documents...")
    documents = ingest_documents()

    if not documents:
        print("❌ No documents found. Please add documents to the docs directory.")
        return

    print("✂️ Chunking...")
    chunks = chunk_documents(documents)

    print("🧠 Generating embeddings...")
    embeddings = embed_chunks(chunks)

    print("📦 Creating FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    print("🔤 Creating SQLite FTS5 index...")
    build_fts_index(chunks, fts_path)

    print("💾 Saving...")
    faiss.write_index(index, str(index_path))
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ Indexing complete: {len(chunks)} chunks indexed")
    print(f"   Index saved to: {index_path}")
    print(f"   Chunks saved to: {chunks_path}")
    print(f"   FTS5 index saved to: {fts_path}")


if __name__ == "__main__":
    build_index()
