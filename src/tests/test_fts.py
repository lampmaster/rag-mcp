"""SQLite FTS5 lexical retrieval."""

import pytest

from rag.chunk import make_chunk_id
from rag.fts import (
    FTSUnavailableError,
    build_fts_index,
    build_fts_query,
    fts_index_exists,
    fts_search,
    tokenize_query,
)

DOCS = {
    "docs/docker.md": (
        "# Docker\n"
        "Local development runs through docker-compose. Start the stack with "
        "`docker compose up -d`. The daemon endpoint is configured through the "
        "DOCKER_HOST environment variable, and images are built from the Dockerfile "
        "in the repository root."
    ),
    "docs/security.md": (
        "# Security\n"
        "Every request to the internal API must carry a Bearer credential in the "
        "Authorization header. The AUTH_TOKEN is issued by the identity service and "
        "signed with JWT_SECRET, which is rotated every 30 days."
    ),
    "docs/database.md": (
        "# Database\n"
        "The service talks to PostgreSQL. Connection settings live in DATABASE_URL, "
        "and the size of the connection pool is controlled by DB_POOL_SIZE. Increase "
        "it carefully: every worker keeps its own pool."
    ),
    "docs/kubernetes.md": (
        "# Kubernetes\n"
        "Deployments are applied with kubectl apply -f. All workloads live in the "
        "namespace configured through K8S_NAMESPACE, and configuration is mounted "
        "from a ConfigMap."
    ),
}


@pytest.fixture
def fts_db(tmp_path):
    chunks = [
        {
            "chunk_id": make_chunk_id(source, 0),
            "source": source,
            "text": text,
            "chunk_index": 0,
        }
        for source, text in DOCS.items()
    ]
    db_path = tmp_path / "fts_index.db"
    build_fts_index(chunks, db_path)
    return db_path


def sources_of(results):
    return [r.source for r in results]


# 8. FTS exact filename retrieval ----------------------------------------------

def test_fts_finds_exact_filename(fts_db):
    results = fts_search("docker.md", 5, fts_db)
    assert sources_of(results)[0] == "docs/docker.md"


def test_fts_finds_filename_inside_a_noisy_question(fts_db):
    query = build_fts_query(
        "Hi, my name is Misha! I want to know where the docker.md file is",
        keywords=["docker.md"],
    )
    results = fts_search(query, 5, fts_db)
    assert sources_of(results)[0] == "docs/docker.md"


# 9. FTS environment variable retrieval ----------------------------------------

@pytest.mark.parametrize(
    "term, expected",
    [
        ("AUTH_TOKEN", "docs/security.md"),
        ("JWT_SECRET", "docs/security.md"),
        ("DB_POOL_SIZE", "docs/database.md"),
        ("DATABASE_URL", "docs/database.md"),
        ("K8S_NAMESPACE", "docs/kubernetes.md"),
        ("DOCKER_HOST", "docs/docker.md"),
    ],
)
def test_fts_finds_environment_variables(fts_db, term, expected):
    results = fts_search(term, 5, fts_db)
    assert results, f"{term} was not retrieved at all"
    assert results[0].source == expected


def test_fts_is_case_insensitive(fts_db):
    assert sources_of(fts_search("auth_token", 5, fts_db))[0] == "docs/security.md"


def test_fts_finds_commands(fts_db):
    assert sources_of(fts_search("kubectl apply", 5, fts_db))[0] == "docs/kubernetes.md"
    assert sources_of(fts_search("docker-compose", 5, fts_db))[0] == "docs/docker.md"


def test_fts_results_are_ranked_from_one(fts_db):
    results = fts_search("AUTH_TOKEN Bearer JWT_SECRET", 5, fts_db)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    assert all(r.retriever == "fts" for r in results)
    assert all(r.chunk_id for r in results)


def test_fts_chunk_ids_match_the_indexed_chunks(fts_db):
    results = fts_search("AUTH_TOKEN", 5, fts_db)
    assert results[0].chunk_id == make_chunk_id("docs/security.md", 0)


def test_fts_empty_query_returns_nothing(fts_db):
    assert fts_search("   ", 5, fts_db) == []


def test_fts_query_syntax_characters_do_not_crash(fts_db):
    # Unbalanced quotes / FTS5 operators must not raise a syntax error.
    assert isinstance(fts_search('AUTH_TOKEN OR " NEAR(', 5, fts_db), list)


def test_missing_index_raises_unavailable(tmp_path):
    with pytest.raises(FTSUnavailableError):
        fts_search("anything", 5, tmp_path / "missing.db")


def test_fts_index_exists(fts_db, tmp_path):
    assert fts_index_exists(fts_db)
    assert not fts_index_exists(tmp_path / "missing.db")


def test_build_fts_query_prioritises_keywords_and_dedupes():
    expr = build_fts_query("where is docker.md", keywords=["docker.md"], max_terms=3)
    assert expr.startswith('"docker.md" *')
    assert expr.count("docker.md") == 1
    assert expr.split(" OR ").__len__() == 3


def test_tokenize_keeps_technical_terms_intact():
    terms = tokenize_query("Set AUTH_TOKEN in docker-compose.yml, see docs/docker.md.")
    assert "AUTH_TOKEN" in terms
    assert "docker-compose.yml" in terms
    assert "docs/docker.md" in terms
