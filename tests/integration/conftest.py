"""Shared fixtures for integration tests that use a real pgvector PostgreSQL container.

Equivalent to mongodb-memory-server in Node.js: spins up a real, ephemeral
PostgreSQL instance with pgvector for production-parity testing.
"""

import hashlib
from typing import List

import pytest
import sqlalchemy
from sqlalchemy import text
from sqlalchemy.orm import Session
from testcontainers.postgres import PostgresContainer

from app.services.vector_store.async_pg_vector import AsyncPgVector
from app.services.vector_store.factory import get_vector_store
from tests.conftest import ORIGINAL_PGVECTOR_POST_INIT

PGVECTOR_IMAGE = "pgvector/pgvector:pg16"

# The shared table is created with `vector(3)`, so every store built against
# this container embeds into three dimensions.
TEST_DIMENSIONS = 3


@pytest.fixture(scope="session")
def pg_container():
    """Start a pgvector PostgreSQL container once for the entire test session.

    Skips — rather than errors — when no container runtime is reachable, so the
    DB-dependent suites stay runnable on machines without Docker.
    """
    container = PostgresContainer(PGVECTOR_IMAGE, driver="psycopg2")
    try:
        container.start()
    except Exception as exc:
        pytest.skip(
            f"PostgreSQL is unreachable ({type(exc).__name__}: {exc}); "
            "skipping database-dependent tests"
        )
    try:
        yield container
    finally:
        container.stop()


@pytest.fixture(scope="session")
def pg_url(pg_container):
    """SQLAlchemy connection URL for the test container."""
    return pg_container.get_connection_url()


@pytest.fixture(scope="session")
def engine(pg_url):
    """Session-scoped SQLAlchemy engine connected to the test container."""
    eng = sqlalchemy.create_engine(pg_url)
    with eng.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    yield eng
    eng.dispose()


@pytest.fixture(scope="session")
def _create_tables(engine):
    """Create the langchain tables and indexes once for the session.

    Mirrors the production schema created by LangChain PGVector + our
    ensure_vector_indexes() startup logic.
    """
    with engine.begin() as conn:
        conn.execute(
            text(
                """
            CREATE TABLE IF NOT EXISTS langchain_pg_collection (
                uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name VARCHAR NOT NULL UNIQUE,
                cmetadata JSONB
            )
        """
            )
        )
        conn.execute(
            text(
                """
            CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
                uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                collection_id UUID REFERENCES langchain_pg_collection(uuid) ON DELETE CASCADE,
                embedding vector(3),
                document VARCHAR,
                cmetadata JSONB,
                custom_id VARCHAR
            )
        """
            )
        )
        conn.execute(
            text(
                """
            CREATE INDEX IF NOT EXISTS idx_langchain_pg_embedding_file_id
            ON langchain_pg_embedding ((cmetadata->>'file_id'))
        """
            )
        )
        conn.execute(
            text(
                """
            CREATE INDEX IF NOT EXISTS ix_cmetadata_gin
            ON langchain_pg_embedding
            USING gin (cmetadata jsonb_path_ops)
        """
            )
        )


@pytest.fixture(scope="session")
def collection_id(engine, _create_tables):
    """Insert a test collection and return its UUID."""
    with engine.begin() as conn:
        row = conn.execute(
            text(
                "INSERT INTO langchain_pg_collection (name, cmetadata) "
                "VALUES (:name, :meta) RETURNING uuid"
            ),
            {"name": "test_collection", "meta": "{}"},
        ).fetchone()
    return row[0]


class DeterministicEmbeddings:
    """Three-dimensional embeddings derived from the text, plus a call log.

    Real inference is the only thing faked here; every store write, SQL
    predicate and cosine computation below runs for real against PostgreSQL.
    """

    def __init__(self, vectors=None):
        self.vectors = vectors or {}
        self.queries: List[str] = []
        self.documents: List[str] = []

    def _vector(self, text_value: str) -> List[float]:
        if text_value in self.vectors:
            return list(self.vectors[text_value])
        digest = hashlib.sha256(text_value.encode("utf-8")).digest()
        return [(digest[index] / 255.0) + 0.01 for index in range(TEST_DIMENSIONS)]

    def embed_query(self, text_value: str) -> List[float]:
        self.queries.append(text_value)
        return self._vector(text_value)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.documents.extend(texts)
        return [self._vector(text_value) for text_value in texts]


@pytest.fixture()
def pg_store(pg_url, engine, _create_tables, monkeypatch):
    """A real AsyncPgVector bound to the container, with its own collection."""
    monkeypatch.setattr(AsyncPgVector, "__post_init__", ORIGINAL_PGVECTOR_POST_INIT)
    embeddings = DeterministicEmbeddings()
    store = get_vector_store(
        connection_string=pg_url,
        embeddings=embeddings,
        collection_name=f"integration-{id(embeddings)}",
        mode="async",
        create_extension=False,
    )
    store.embeddings_log = embeddings
    yield store
    store._bind.dispose()


@pytest.fixture()
def db_session(engine, _create_tables):
    """Per-test session that rolls back after each test for isolation."""
    conn = engine.connect()
    trans = conn.begin()
    session = Session(bind=conn)
    yield session, conn
    session.close()
    trans.rollback()
    conn.close()
