import asyncio

import pytest
from app.services.database import ensure_vector_indexes, PSQLDatabase


class CapturingConnection:
    """Records every SQL statement passed to execute()."""

    def __init__(self):
        self.statements = []

    async def fetchval(self, query, index_name):
        return False

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


def _run_with_captured_conn(monkeypatch):
    """Run ensure_vector_indexes() and return the captured connection."""
    conn = CapturingConnection()
    pool = CapturingPool(conn)

    async def fake_get_pool():
        return pool

    monkeypatch.setattr(PSQLDatabase, "get_pool", fake_get_pool)
    asyncio.run(ensure_vector_indexes())
    return conn


def test_ensure_vector_indexes(monkeypatch):
    conn = _run_with_captured_conn(monkeypatch)
    assert len(conn.statements) > 0


def test_ensure_vector_indexes_do_block_dollar_quoting(monkeypatch):
    """DO block must use $$ dollar-quoting, not single $."""
    conn = _run_with_captured_conn(monkeypatch)
    do_block = next(s for s in conn.statements if "DO" in s)
    assert "$$" in do_block, "DO block must use $$ dollar-quoting"


def test_ensure_vector_indexes_jsonb_migration_sql(monkeypatch):
    """Migration block contains the correct ALTER COLUMN and schema filter."""
    conn = _run_with_captured_conn(monkeypatch)
    do_block = next(s for s in conn.statements if "DO" in s)
    assert "TYPE JSONB" in do_block
    assert "cmetadata::jsonb" in do_block
    assert "table_schema = current_schema()" in do_block


def test_ensure_vector_indexes_lock_timeout(monkeypatch):
    """Migration sets a lock_timeout before ALTER TABLE."""
    conn = _run_with_captured_conn(monkeypatch)
    do_block = next(s for s in conn.statements if "DO" in s)
    assert "lock_timeout" in do_block


def test_ensure_vector_indexes_gin_index(monkeypatch):
    """GIN index with jsonb_path_ops is created."""
    conn = _run_with_captured_conn(monkeypatch)
    gin_stmt = next(s for s in conn.statements if "ix_cmetadata_gin" in s)
    assert "jsonb_path_ops" in gin_stmt
    assert "USING gin" in gin_stmt


def _capture_create_pool_kwargs(monkeypatch, schema):
    """Reset the pool, set POSTGRES_SCHEMA, and return the kwargs that
    PSQLDatabase.get_pool() passes to asyncpg.create_pool()."""
    import app.services.database as db

    captured = {}

    async def fake_create_pool(**kwargs):
        captured.update(kwargs)
        return object()  # sentinel pool

    monkeypatch.setattr(db.asyncpg, "create_pool", fake_create_pool)
    monkeypatch.setattr(db, "POSTGRES_SCHEMA", schema)
    monkeypatch.setattr(db.PSQLDatabase, "pool", None)

    asyncio.run(db.PSQLDatabase.get_pool())
    return captured


def test_get_pool_pins_search_path_when_schema_configured(monkeypatch):
    """When POSTGRES_SCHEMA is set the asyncpg pool must receive a matching
    search_path via server_settings (regression: ensure_vector_indexes ran in
    the wrong schema because the pool used the bare DSN)."""
    captured = _capture_create_pool_kwargs(monkeypatch, "myapp")
    assert captured["server_settings"] == {"search_path": "myapp,public"}


def test_get_pool_search_path_multiple_schemas(monkeypatch):
    """Comma-separated POSTGRES_SCHEMA is honored and `public` is appended,
    matching the SQLAlchemy engine's _build_search_path."""
    captured = _capture_create_pool_kwargs(monkeypatch, "myapp, extensions")
    assert captured["server_settings"] == {"search_path": "myapp,extensions,public"}


def test_get_pool_no_search_path_without_schema(monkeypatch):
    """With no POSTGRES_SCHEMA the pool keeps the default search_path (no
    server_settings), preserving today's behavior."""
    captured = _capture_create_pool_kwargs(monkeypatch, None)
    assert "server_settings" not in captured
