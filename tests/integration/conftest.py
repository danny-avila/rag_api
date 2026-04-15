"""Shared fixtures for integration tests that use a real pgvector PostgreSQL container.

Equivalent to mongodb-memory-server in Node.js: spins up a real, ephemeral
PostgreSQL instance with pgvector for production-parity testing.
"""

import pytest
import sqlalchemy
from sqlalchemy import text
from sqlalchemy.orm import Session
from testcontainers.postgres import PostgresContainer

PGVECTOR_IMAGE = "pgvector/pgvector:pg16"


@pytest.fixture(scope="session")
def pg_container():
    """Start a pgvector PostgreSQL container once for the entire test session."""
    with PostgresContainer(PGVECTOR_IMAGE, driver="psycopg2") as pg:
        yield pg


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
