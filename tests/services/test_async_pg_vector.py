import asyncio
from unittest.mock import patch, MagicMock
import pytest
from langchain_core.documents import Document
from app.services.vector_store.async_pg_vector import AsyncPgVector
from app.services.vector_store.extended_pg_vector import ExtendedPgVector


class DummyAsyncPgVector(AsyncPgVector):
    """Subclass that skips DB initialization."""

    def __init__(self):
        # Bypass ExtendedPgVector/PGVector __init__ entirely
        self._thread_pool = None
        self._bind = None  # Prevent AttributeError in PGVector.__del__


@pytest.fixture
def store():
    return DummyAsyncPgVector()


@pytest.mark.asyncio
async def test_get_all_ids_dispatches_to_super(store):
    with patch.object(
        ExtendedPgVector, "get_all_ids", return_value=["id1", "id2"]
    ) as mock:
        result = await store.get_all_ids()
    mock.assert_called_once_with()
    assert result == ["id1", "id2"]


@pytest.mark.asyncio
async def test_get_filtered_ids_passes_ids(store):
    with patch.object(
        ExtendedPgVector, "get_filtered_ids", return_value=["id1"]
    ) as mock:
        result = await store.get_filtered_ids(["id1", "id2"])
    mock.assert_called_once_with(["id1", "id2"])
    assert result == ["id1"]


@pytest.mark.asyncio
async def test_get_documents_by_ids_passes_ids(store):
    docs = [Document(page_content="test", metadata={"file_id": "id1"})]
    with patch.object(
        ExtendedPgVector, "get_documents_by_ids", return_value=docs
    ) as mock:
        result = await store.get_documents_by_ids(["id1"])
    mock.assert_called_once_with(["id1"])
    assert result == docs


@pytest.mark.asyncio
async def test_delete_passes_args(store):
    with patch.object(ExtendedPgVector, "_delete_multiple") as mock:
        await store.delete(ids=["id1"], collection_only=True)
    mock.assert_called_once_with(["id1"], True)


@pytest.mark.asyncio
async def test_asimilarity_search_passes_args(store):
    expected = [(Document(page_content="test", metadata={}), 0.9)]
    with patch.object(
        ExtendedPgVector,
        "similarity_search_with_score_by_vector",
        return_value=expected,
    ) as mock:
        embedding = [0.1, 0.2, 0.3]
        result = await store.asimilarity_search_with_score_by_vector(
            embedding, k=5, filter={"file_id": {"$eq": "id1"}}
        )
    mock.assert_called_once_with(embedding, 5, {"file_id": {"$eq": "id1"}})
    assert result == expected


@pytest.mark.asyncio
async def test_aadd_documents_passes_args(store):
    docs = [Document(page_content="test", metadata={})]
    with patch.object(ExtendedPgVector, "add_documents", return_value=["id1"]) as mock:
        result = await store.aadd_documents(docs, ids=["id1"])
    mock.assert_called_once_with(docs, ids=["id1"])
    assert result == ["id1"]


@pytest.mark.asyncio
async def test_run_in_executor_converts_stop_iteration(store):
    """StopIteration can't be set on an asyncio.Future — verify it becomes RuntimeError."""

    def raises_stop():
        raise StopIteration("exhausted")

    with patch.object(ExtendedPgVector, "get_all_ids", side_effect=raises_stop):
        with pytest.raises(RuntimeError):
            await store.get_all_ids()
