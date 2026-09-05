"""Stable chunk ids shared by the vector metadata and the FTS5 index."""

import sqlite3

from rag.chunk import chunk_documents, make_chunk_id
from rag.fts import FTS_TABLE, build_fts_index, fts_search

DOC = {
    "path": "docs/docker.md",
    "text": "Start the stack with docker compose up. " * 200,
}


def test_chunk_ids_are_stable_across_runs():
    first = chunk_documents([DOC])
    second = chunk_documents([DOC])

    assert len(first) > 1, "the test document must produce several chunks"
    assert [c["chunk_id"] for c in first] == [c["chunk_id"] for c in second]
    assert first[0]["chunk_id"] == make_chunk_id("docs/docker.md", 0)


def test_chunk_ids_are_unique_across_documents():
    other = {"path": "docs/security.md", "text": DOC["text"]}
    chunks = chunk_documents([DOC, other])
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_vector_metadata_and_fts_share_the_same_chunk_ids(tmp_path):
    chunks = chunk_documents([DOC])          # chunked exactly once
    db_path = tmp_path / "fts.db"
    build_fts_index(chunks, db_path)          # same chunks feed FTS5

    conn = sqlite3.connect(str(db_path))
    try:
        stored = {row[0] for row in conn.execute(f"SELECT chunk_id FROM {FTS_TABLE}")}
    finally:
        conn.close()

    assert stored == {c["chunk_id"] for c in chunks}

    hit = fts_search("docker compose up", 1, db_path)[0]
    assert hit.chunk_id in stored
    assert hit.source == "docs/docker.md"
