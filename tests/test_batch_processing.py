# tests/test_batch_processing.py
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from langchain_core.documents import Document


class TestBatchProcessing:
    """Test batch processing functions."""

    @pytest.fixture
    def mock_documents(self):
        """Create mock documents for testing."""
        return [
            Document(page_content=f"content_{i}", metadata={"file_id": "test_file"})
            for i in range(10)
        ]

    @pytest.fixture
    def mock_async_vector_store(self):
        """Create mock async vector store."""
        store = AsyncMock()
        store.aadd_documents = AsyncMock(return_value=["id1", "id2"])
        store.delete = AsyncMock()
        return store

    @pytest.fixture
    def mock_sync_vector_store(self):
        """Create mock sync vector store."""
        store = Mock()
        store.add_documents = Mock(return_value=["id1", "id2"])
        store.delete = Mock()
        return store

    # --- Async Pipeline Tests ---

    @pytest.mark.asyncio
    async def test_async_pipeline_basic(self, mock_documents, mock_async_vector_store):
        """Test basic async pipeline processing."""
        from app.routes.document_routes import _process_documents_async_pipeline

        with patch("app.routes.document_routes.EMBEDDING_BATCH_SIZE", 3):
            result = await _process_documents_async_pipeline(
                documents=mock_documents,
                file_id="test_file",
                vector_store=mock_async_vector_store,
                executor=None,
            )

        assert len(result) > 0
        assert mock_async_vector_store.aadd_documents.called

    @pytest.mark.asyncio
    async def test_async_pipeline_single_batch(self, mock_async_vector_store):
        """Test when all documents fit in one batch."""
        from app.routes.document_routes import _process_documents_async_pipeline

        docs = [Document(page_content="test", metadata={})]
        with patch("app.routes.document_routes.EMBEDDING_BATCH_SIZE", 10):
            result = await _process_documents_async_pipeline(
                documents=docs,
                file_id="test_file",
                vector_store=mock_async_vector_store,
                executor=None,
            )

        assert mock_async_vector_store.aadd_documents.call_count == 1

    @pytest.mark.asyncio
    async def test_async_pipeline_exact_batch_size(self, mock_async_vector_store):
        """Test when document count equals batch size."""
        from app.routes.document_routes import _process_documents_async_pipeline

        docs = [Document(page_content=f"test_{i}", metadata={}) for i in range(5)]
        with patch("app.routes.document_routes.EMBEDDING_BATCH_SIZE", 5):
            result = await _process_documents_async_pipeline(
                documents=docs,
                file_id="test_file",
                vector_store=mock_async_vector_store,
                executor=None,
            )

        assert mock_async_vector_store.aadd_documents.call_count == 1

    @pytest.mark.asyncio
    async def test_async_pipeline_empty_documents(self, mock_async_vector_store):
        """Test with empty document list."""
        from app.routes.document_routes import _process_documents_async_pipeline

        with patch("app.routes.document_routes.EMBEDDING_BATCH_SIZE", 3):
            result = await _process_documents_async_pipeline(
                documents=[],
                file_id="test_file",
                vector_store=mock_async_vector_store,
                executor=None,
            )

        assert result == []
        assert not mock_async_vector_store.aadd_documents.called

    @pytest.mark.asyncio
    async def test_async_pipeline_rollback_on_error(
        self, mock_documents, mock_async_vector_store
    ):
        """Test that rollback occurs when insertion fails after some success."""
        from app.routes.document_routes import _process_documents_async_pipeline

        # First batch succeeds, second batch fails
        mock_async_vector_store.aadd_documents = AsyncMock(
            side_effect=[["id1"], Exception("DB error")]
        )

        with patch("app.routes.document_routes.EMBEDDING_BATCH_SIZE", 3):
            with pytest.raises(Exception, match="DB error"):
                await _process_documents_async_pipeline(
                    documents=mock_documents,
                    file_id="test_file",
                    vector_store=mock_async_vector_store,
                    executor=None,
                )

        mock_async_vector_store.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_pipeline_no_rollback_on_first_batch_error(
        self, mock_documents, mock_async_vector_store
    ):
        """Test that no rollback occurs if first batch fails (nothing inserted)."""
        from app.routes.document_routes import _process_documents_async_pipeline

        mock_async_vector_store.aadd_documents = AsyncMock(
            side_effect=Exception("DB error")
        )

        with patch("app.routes.document_routes.EMBEDDING_BATCH_SIZE", 3):
            with pytest.raises(Exception):
                await _process_documents_async_pipeline(
                    documents=mock_documents,
                    file_id="test_file",
                    vector_store=mock_async_vector_store,
                    executor=None,
                )

        # Should not attempt rollback since nothing was inserted
        assert not mock_async_vector_store.delete.called

    # --- Sync Batched Tests ---

    @pytest.mark.asyncio
    async def test_sync_batched_basic(self, mock_documents, mock_sync_vector_store):
        """Test basic sync batch processing."""
        from app.routes.document_routes import _process_documents_batched_sync
        import asyncio

        # Create a real executor for the test
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2) as executor:
            with patch("app.routes.document_routes.EMBEDDING_BATCH_SIZE", 3):
                result = await _process_documents_batched_sync(
                    documents=mock_documents,
                    file_id="test_file",
                    vector_store=mock_sync_vector_store,
                    executor=executor,
                )

        assert len(result) > 0
        assert mock_sync_vector_store.add_documents.called

    @pytest.mark.asyncio
    async def test_sync_batched_empty_documents(self, mock_sync_vector_store):
        """Test sync batch processing with empty documents."""
        from app.routes.document_routes import _process_documents_batched_sync

        with patch("app.routes.document_routes.EMBEDDING_BATCH_SIZE", 3):
            result = await _process_documents_batched_sync(
                documents=[],
                file_id="test_file",
                vector_store=mock_sync_vector_store,
                executor=None,
            )

        assert result == []
        assert not mock_sync_vector_store.add_documents.called

    @pytest.mark.asyncio
    async def test_sync_batched_rollback_on_error(
        self, mock_documents, mock_sync_vector_store
    ):
        """Test sync rollback behavior."""
        from app.routes.document_routes import _process_documents_batched_sync
        from concurrent.futures import ThreadPoolExecutor

        # First batch succeeds, second batch fails
        mock_sync_vector_store.add_documents = Mock(
            side_effect=[["id1"], Exception("DB error")]
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            with patch("app.routes.document_routes.EMBEDDING_BATCH_SIZE", 3):
                with pytest.raises(Exception):
                    await _process_documents_batched_sync(
                        documents=mock_documents,
                        file_id="test_file",
                        vector_store=mock_sync_vector_store,
                        executor=executor,
                    )

        mock_sync_vector_store.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_batched_no_rollback_on_first_error(
        self, mock_documents, mock_sync_vector_store
    ):
        """Test that no rollback occurs if first batch fails."""
        from app.routes.document_routes import _process_documents_batched_sync
        from concurrent.futures import ThreadPoolExecutor

        mock_sync_vector_store.add_documents = Mock(side_effect=Exception("DB error"))

        with ThreadPoolExecutor(max_workers=2) as executor:
            with patch("app.routes.document_routes.EMBEDDING_BATCH_SIZE", 3):
                with pytest.raises(Exception):
                    await _process_documents_batched_sync(
                        documents=mock_documents,
                        file_id="test_file",
                        vector_store=mock_sync_vector_store,
                        executor=executor,
                    )

        # Should not attempt rollback since nothing was inserted
        assert not mock_sync_vector_store.delete.called


class TestBatchConfiguration:
    """Test configuration and edge cases."""

    def test_batch_calculation(self):
        """Test batch count calculation using the utility function."""
        from app.routes.document_routes import calculate_num_batches

        # 10 docs, batch size 3 = 4 batches (3+3+3+1)
        assert calculate_num_batches(10, 3) == 4

        # Exact division
        assert calculate_num_batches(9, 3) == 3

        # Single item
        assert calculate_num_batches(1, 3) == 1

        # Zero items
        assert calculate_num_batches(0, 3) == 0

        # Batch size larger than total
        assert calculate_num_batches(5, 10) == 1

        # Edge case: batch_size of 0 returns 1 (fallback)
        assert calculate_num_batches(10, 0) == 1

        # Edge case: batch_size of 1
        assert calculate_num_batches(5, 1) == 5

    def test_embedding_batch_size_from_env(self):
        """Test that EMBEDDING_BATCH_SIZE is read from environment variable."""
        import os
        from importlib import reload

        # Save current value
        original = os.environ.get("EMBEDDING_BATCH_SIZE")

        try:
            # Set a specific test value
            os.environ["EMBEDDING_BATCH_SIZE"] = "999"

            import app.config as config_module

            reload(config_module)

            assert config_module.EMBEDDING_BATCH_SIZE == 999
        finally:
            # Restore original value
            if original is not None:
                os.environ["EMBEDDING_BATCH_SIZE"] = original
            elif "EMBEDDING_BATCH_SIZE" in os.environ:
                del os.environ["EMBEDDING_BATCH_SIZE"]

    def test_embedding_max_queue_size_from_env(self):
        """Test that EMBEDDING_MAX_QUEUE_SIZE is read from environment variable."""
        import os
        from importlib import reload

        original = os.environ.get("EMBEDDING_MAX_QUEUE_SIZE")

        try:
            os.environ["EMBEDDING_MAX_QUEUE_SIZE"] = "10"

            import app.config as config_module

            reload(config_module)

            assert config_module.EMBEDDING_MAX_QUEUE_SIZE == 10
        finally:
            if original is not None:
                os.environ["EMBEDDING_MAX_QUEUE_SIZE"] = original
            elif "EMBEDDING_MAX_QUEUE_SIZE" in os.environ:
                del os.environ["EMBEDDING_MAX_QUEUE_SIZE"]


class TestBatchSizeEdgeCases:
    """Test various batch size configurations."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "batch_size,doc_count,expected_batches",
        [
            (1, 5, 5),  # Each doc separate
            (5, 5, 1),  # Exact fit
            (10, 5, 1),  # Batch larger than docs
            (3, 10, 4),  # Normal case
            (100, 1, 1),  # Large batch, single doc
        ],
    )
    async def test_batch_counts(self, batch_size, doc_count, expected_batches):
        """Test various batch size and document count combinations."""
        from app.routes.document_routes import _process_documents_async_pipeline

        mock_store = AsyncMock()
        mock_store.aadd_documents = AsyncMock(return_value=["id"])

        docs = [
            Document(page_content=f"doc_{i}", metadata={}) for i in range(doc_count)
        ]

        with patch("app.routes.document_routes.EMBEDDING_BATCH_SIZE", batch_size):
            await _process_documents_async_pipeline(
                documents=docs, file_id="test", vector_store=mock_store, executor=None
            )

        assert mock_store.aadd_documents.call_count == expected_batches

    @pytest.mark.asyncio
    async def test_large_batch_size_single_call(self):
        """Test that a very large batch size results in a single call."""
        from app.routes.document_routes import _process_documents_async_pipeline

        mock_store = AsyncMock()
        mock_store.aadd_documents = AsyncMock(return_value=["id"])

        docs = [Document(page_content=f"doc_{i}", metadata={}) for i in range(100)]

        with patch("app.routes.document_routes.EMBEDDING_BATCH_SIZE", 1000):
            await _process_documents_async_pipeline(
                documents=docs, file_id="test", vector_store=mock_store, executor=None
            )

        assert mock_store.aadd_documents.call_count == 1

    @pytest.mark.asyncio
    async def test_batch_size_one_multiple_calls(self):
        """Test that batch size of 1 results in many calls."""
        from app.routes.document_routes import _process_documents_async_pipeline

        mock_store = AsyncMock()
        mock_store.aadd_documents = AsyncMock(return_value=["id"])

        docs = [Document(page_content=f"doc_{i}", metadata={}) for i in range(5)]

        with patch("app.routes.document_routes.EMBEDDING_BATCH_SIZE", 1):
            await _process_documents_async_pipeline(
                documents=docs, file_id="test", vector_store=mock_store, executor=None
            )

        assert mock_store.aadd_documents.call_count == 5


class TestProducerConsumerPattern:
    """Test the producer-consumer pattern behavior."""

    @pytest.mark.asyncio
    async def test_producer_signals_completion_on_success(self):
        """Test that producer always signals completion."""
        from app.routes.document_routes import _process_documents_async_pipeline

        mock_store = AsyncMock()
        mock_store.aadd_documents = AsyncMock(return_value=["id"])

        docs = [Document(page_content="test", metadata={})]

        with patch("app.routes.document_routes.EMBEDDING_BATCH_SIZE", 1):
            result = await _process_documents_async_pipeline(
                documents=docs, file_id="test", vector_store=mock_store, executor=None
            )

        # If we get here without hanging, the producer signaled completion
        assert result == ["id"]

    @pytest.mark.asyncio
    async def test_consumer_handles_exception_in_batch(self):
        """Test that consumer properly handles exceptions from vector store."""
        from app.routes.document_routes import _process_documents_async_pipeline

        mock_store = AsyncMock()
        mock_store.aadd_documents = AsyncMock(side_effect=ValueError("Test error"))
        mock_store.delete = AsyncMock()

        docs = [Document(page_content="test", metadata={})]

        with patch("app.routes.document_routes.EMBEDDING_BATCH_SIZE", 1):
            with pytest.raises(ValueError, match="Test error"):
                await _process_documents_async_pipeline(
                    documents=docs,
                    file_id="test",
                    vector_store=mock_store,
                    executor=None,
                )

    @pytest.mark.asyncio
    async def test_all_ids_collected_across_batches(self):
        """Test that IDs from all batches are collected."""
        from app.routes.document_routes import _process_documents_async_pipeline

        mock_store = AsyncMock()
        # Return different IDs for each batch
        mock_store.aadd_documents = AsyncMock(
            side_effect=[["id1", "id2"], ["id3", "id4"], ["id5"]]
        )

        docs = [Document(page_content=f"doc_{i}", metadata={}) for i in range(5)]

        with patch("app.routes.document_routes.EMBEDDING_BATCH_SIZE", 2):
            result = await _process_documents_async_pipeline(
                documents=docs, file_id="test", vector_store=mock_store, executor=None
            )

        assert len(result) == 5
        assert result == ["id1", "id2", "id3", "id4", "id5"]
