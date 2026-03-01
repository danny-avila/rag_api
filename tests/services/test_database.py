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
