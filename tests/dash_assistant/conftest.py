# tests/dash_assistant/conftest.py
"""Shared fixtures for dash assistant tests."""
import pytest
import pytest_asyncio
import asyncpg
from pathlib import Path
from app.config import DSN
from app.dash_assistant.db import DashAssistantDB


@pytest_asyncio.fixture(scope="session")
async def db_pool():
    """Create a database pool for the entire test session."""
    pool = await asyncpg.create_pool(dsn=DSN, min_size=1, max_size=5)
    yield pool
    await pool.close()


@pytest_asyncio.fixture
async def db_connection(db_pool):
    """Get a database connection from the pool."""
    async with db_pool.acquire() as conn:
        yield conn


@pytest_asyncio.fixture
async def clean_db(db_connection):
    """Clean database before and after each test."""
    # Clean up before test
    await db_connection.execute("DELETE FROM bi_chunk")
    await db_connection.execute("DELETE FROM bi_chart") 
    await db_connection.execute("DELETE FROM bi_entity")
    
    yield db_connection
    
    # Clean up after test
    await db_connection.execute("DELETE FROM bi_chunk")
    await db_connection.execute("DELETE FROM bi_chart")
    await db_connection.execute("DELETE FROM bi_entity")


@pytest.fixture
def fixtures_dir():
    """Get path to test fixtures directory."""
    return Path(__file__).parent.parent / "fixtures" / "superset"


@pytest_asyncio.fixture
async def mock_index_job(db_pool, monkeypatch):
    """Create IndexJob with mocked database pool."""
    # Mock the DashAssistantDB to use our test pool
    async def mock_get_pool():
        return db_pool
    
    async def mock_execute_query(query, *args):
        async with db_pool.acquire() as conn:
            await conn.execute(query, *args)
    
    async def mock_fetch_one(query, *args):
        async with db_pool.acquire() as conn:
            return await conn.fetchrow(query, *args)
    
    async def mock_fetch_all(query, *args):
        async with db_pool.acquire() as conn:
            return await conn.fetch(query, *args)
    
    async def mock_fetch_value(query, *args):
        async with db_pool.acquire() as conn:
            return await conn.fetchval(query, *args)
    
    # Patch DashAssistantDB methods
    monkeypatch.setattr(DashAssistantDB, "get_pool", mock_get_pool)
    monkeypatch.setattr(DashAssistantDB, "execute_query", mock_execute_query)
    monkeypatch.setattr(DashAssistantDB, "fetch_one", mock_fetch_one)
    monkeypatch.setattr(DashAssistantDB, "fetch_all", mock_fetch_all)
    monkeypatch.setattr(DashAssistantDB, "fetch_value", mock_fetch_value)
    
    # Don't close pool in tests
    async def mock_close_pool():
        pass
    monkeypatch.setattr(DashAssistantDB, "close_pool", mock_close_pool)
    
    from app.dash_assistant.ingestion.index_jobs import IndexJob
    return IndexJob()


@pytest.fixture(autouse=True)
def isolate_tests():
    """Ensure test isolation by resetting any global state."""
    # Reset any global state here if needed
    yield
    # Cleanup after test
