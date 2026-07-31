import asyncio
from unittest.mock import AsyncMock

import pytest
from app.services import database
from app.services.database import PSQLDatabase, ensure_vector_indexes


class CapturingConnection:
    """Records every SQL statement passed to execute()."""

    def __init__(self, cmetadata_type="jsonb"):
        self.statements = []
        self.queries = []
        self.cmetadata_type = cmetadata_type

    async def fetchval(self, query):
        self.queries.append(query)
        return self.cmetadata_type

    async def execute(self, query):
        self.statements.append(query)
        return "Executed"


class CapturingAcquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        pass


class CapturingPool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return CapturingAcquire(self._conn)


DDL_FLAGS = (
    "PGVECTOR_CREATE_LEGACY_INDEXES",
    "PGVECTOR_MIGRATE_CMETADATA_JSONB",
    "PGVECTOR_CREATE_CMETADATA_GIN_INDEX",
)


def test_get_pool_uses_configured_schema_search_path(monkeypatch):
    expected_pool = object()
    create_pool = AsyncMock(return_value=expected_pool)
    monkeypatch.setattr(database.asyncpg, "create_pool", create_pool)
    monkeypatch.setattr(database, "POSTGRES_SCHEMA", "myapp, extensions")
    monkeypatch.setattr(PSQLDatabase, "pool", None)

    pool = asyncio.run(PSQLDatabase.get_pool())

    assert pool is expected_pool
    create_pool.assert_awaited_once_with(
        dsn=database.DSN,
        server_settings={"search_path": "myapp,extensions,public"},
    )


def _run_with_captured_conn(monkeypatch, *enabled_flags, cmetadata_type="jsonb"):
    """Run ensure_vector_indexes() and return the captured connection."""
    for flag in DDL_FLAGS:
        monkeypatch.delenv(flag, raising=False)
    for flag in enabled_flags:
        monkeypatch.setenv(flag, "true")

    conn = CapturingConnection(cmetadata_type=cmetadata_type)
    pool = CapturingPool(conn)

    async def fake_get_pool():
        return pool

    monkeypatch.setattr(PSQLDatabase, "get_pool", fake_get_pool)
    asyncio.run(ensure_vector_indexes())
    return conn


def test_ensure_vector_indexes(monkeypatch):
    conn = _run_with_captured_conn(monkeypatch)
    assert conn.statements == []


def test_ensure_vector_indexes_legacy_indexes_opt_in(monkeypatch):
    conn = _run_with_captured_conn(monkeypatch, "PGVECTOR_CREATE_LEGACY_INDEXES")

    assert len(conn.statements) == 2
    assert "custom_id" in conn.statements[0]
    assert "cmetadata->>'file_id'" in conn.statements[1]


def test_ensure_vector_indexes_do_block_dollar_quoting(monkeypatch):
    """DO block must use $$ dollar-quoting, not single $."""
    conn = _run_with_captured_conn(monkeypatch, "PGVECTOR_MIGRATE_CMETADATA_JSONB")
    do_block = next(s for s in conn.statements if "DO" in s)
    assert "$$" in do_block, "DO block must use $$ dollar-quoting"


def test_ensure_vector_indexes_jsonb_migration_sql(monkeypatch):
    """Migration block contains the correct ALTER COLUMN and schema filter."""
    conn = _run_with_captured_conn(monkeypatch, "PGVECTOR_MIGRATE_CMETADATA_JSONB")
    do_block = next(s for s in conn.statements if "DO" in s)
    assert "TYPE JSONB" in do_block
    assert "cmetadata::jsonb" in do_block
    assert "table_schema = current_schema()" in do_block


def test_ensure_vector_indexes_lock_timeout(monkeypatch):
    """Migration sets a lock_timeout before ALTER TABLE."""
    conn = _run_with_captured_conn(monkeypatch, "PGVECTOR_MIGRATE_CMETADATA_JSONB")
    do_block = next(s for s in conn.statements if "DO" in s)
    assert "lock_timeout" in do_block


def test_ensure_vector_indexes_gin_index(monkeypatch):
    """GIN index with jsonb_path_ops is created."""
    conn = _run_with_captured_conn(monkeypatch, "PGVECTOR_CREATE_CMETADATA_GIN_INDEX")
    gin_stmt = next(s for s in conn.statements if "ix_cmetadata_gin" in s)
    assert "jsonb_path_ops" in gin_stmt
    assert "USING gin" in gin_stmt
    assert any("data_type" in query for query in conn.queries)


def test_ensure_vector_indexes_gin_index_warns_for_legacy_json(monkeypatch, caplog):
    conn = _run_with_captured_conn(
        monkeypatch,
        "PGVECTOR_CREATE_CMETADATA_GIN_INDEX",
        cmetadata_type="json",
    )

    assert conn.statements == []
    assert "uses legacy JSON" in caplog.text
    assert "PGVECTOR_MIGRATE_CMETADATA_JSONB=true" in caplog.text


def test_ensure_vector_indexes_gin_index_warns_for_missing_column(monkeypatch, caplog):
    conn = _run_with_captured_conn(
        monkeypatch,
        "PGVECTOR_CREATE_CMETADATA_GIN_INDEX",
        cmetadata_type=None,
    )

    assert conn.statements == []
    assert "was not found in the current schema" in caplog.text
