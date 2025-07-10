import pytest
from app.services.database import ensure_vector_indexes, PSQLDatabase


# Create dummy classes to simulate a database connection and pool
class DummyConnection:
    async def fetchval(self, query, index_name):
        # Simulate that the index does not exist
        return False

    async def execute(self, query):
        return "Executed"


class DummyAcquire:
    async def __aenter__(self):
        return DummyConnection()

    async def __aexit__(self, exc_type, exc, tb):
        pass


class DummyPool:
    def acquire(self):
        return DummyAcquire()


class DummyDatabase:
    pool = DummyPool()

    @classmethod
    async def get_pool(cls):
        return cls.pool


@pytest.fixture
def dummy_pool(monkeypatch):
    monkeypatch.setattr(PSQLDatabase, "get_pool", DummyDatabase.get_pool)
    return DummyPool()


import asyncio


@pytest.mark.asyncio
async def test_ensure_vector_indexes(monkeypatch, dummy_pool):
    result = await ensure_vector_indexes()
    # If no exceptions are raised, the function worked as expected.
    assert result is None
