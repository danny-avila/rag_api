"""
Unit tests for StorageManager
"""

import pytest
import time
from unittest.mock import Mock, AsyncMock, patch
from app.services.storage.storage_manager import StorageManager


class TestStorageManager:
    """Test StorageManager functionality"""

    def setup_method(self):
        """Setup test instance with mock storage providers"""
        self.mock_primary = Mock()
        self.mock_primary.store_file = AsyncMock()
        self.mock_primary.delete_file = AsyncMock()
        self.mock_primary.get_file_url = Mock()
        self.mock_primary.get_folder_name = Mock()
        self.mock_primary.generate_storage_key = Mock()

        self.mock_fallback = Mock()
        self.mock_fallback.store_file = AsyncMock()
        self.mock_fallback.delete_file = AsyncMock()
        self.mock_fallback.get_file_url = Mock()

        self.manager = StorageManager(
            primary_storage=self.mock_primary,
            fallback_storage=self.mock_fallback,
            circuit_breaker_timeout=5,  # 5 seconds for faster tests
            failure_threshold=2,  # Lower threshold for tests
        )

    @pytest.mark.asyncio
    async def test_store_file_uses_primary_when_available(self):
        """Test that store_file uses primary storage when available"""
        # Mock successful primary storage
        expected_metadata = {"storage_type": "s3", "key": "test.pdf"}
        self.mock_primary.store_file.return_value = expected_metadata

        result = await self.manager.store_file(
            local_file_path="/tmp/test.pdf",
            storage_key="test.pdf",
            content_type="application/pdf",
        )

        assert result == expected_metadata
        self.mock_primary.store_file.assert_called_once()
        self.mock_fallback.store_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_file_falls_back_on_primary_failure(self):
        """Test that store_file falls back when primary fails"""
        # Mock primary failure and fallback success
        self.mock_primary.store_file.side_effect = Exception("S3 error")
        fallback_metadata = {"storage_type": "local", "key": "test.pdf"}
        self.mock_fallback.store_file.return_value = fallback_metadata

        result = await self.manager.store_file(
            local_file_path="/tmp/test.pdf", storage_key="test.pdf"
        )

        assert result == fallback_metadata
        self.mock_primary.store_file.assert_called_once()
        self.mock_fallback.store_file.assert_called_once()
        assert self.manager.primary_failures == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_threshold(self):
        """Test that circuit breaker opens after failure threshold"""
        # Mock repeated primary failures
        self.mock_primary.store_file.side_effect = Exception("S3 error")
        self.mock_fallback.store_file.return_value = {"storage_type": "local"}

        # First failure
        await self.manager.store_file("/tmp/test1.pdf", "test1.pdf")
        assert self.manager.primary_failures == 1
        assert not self.manager.circuit_open

        # Second failure (threshold reached)
        await self.manager.store_file("/tmp/test2.pdf", "test2.pdf")
        assert self.manager.primary_failures == 2
        assert self.manager.circuit_open

        # Third call should skip primary
        self.mock_primary.store_file.reset_mock()
        await self.manager.store_file("/tmp/test3.pdf", "test3.pdf")
        self.mock_primary.store_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets_after_timeout(self):
        """Test that circuit breaker resets after timeout"""
        # Open the circuit
        self.mock_primary.store_file.side_effect = Exception("S3 error")
        self.mock_fallback.store_file.return_value = {"storage_type": "local"}

        # Reach failure threshold
        for _ in range(2):
            await self.manager.store_file("/tmp/test.pdf", "test.pdf")

        assert self.manager.circuit_open

        # Mock time passage
        with patch("time.time") as mock_time:
            # Set current time beyond timeout
            mock_time.return_value = time.time() + 10

            # Reset primary to succeed
            self.mock_primary.store_file.side_effect = None
            self.mock_primary.store_file.return_value = {"storage_type": "s3"}

            # Should retry primary
            result = await self.manager.store_file("/tmp/test.pdf", "test.pdf")

            assert not self.manager.circuit_open
            assert self.manager.primary_failures == 0
            assert result["storage_type"] == "s3"

    @pytest.mark.asyncio
    async def test_store_file_no_fallback_raises_error(self):
        """Test that store_file raises error when no fallback available"""
        # Manager without fallback
        manager = StorageManager(primary_storage=self.mock_primary)

        # Mock primary failure
        self.mock_primary.store_file.side_effect = Exception("S3 error")

        with pytest.raises(Exception) as exc_info:
            await manager.store_file("/tmp/test.pdf", "test.pdf")

        assert "All storage providers failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_delete_file_tries_both_storages(self):
        """Test that delete_file tries both primary and fallback"""
        # Mock both succeed
        self.mock_primary.delete_file.return_value = True
        self.mock_fallback.delete_file.return_value = True

        result = await self.manager.delete_file("test.pdf")

        assert result is True
        self.mock_primary.delete_file.assert_called_once_with("test.pdf")
        self.mock_fallback.delete_file.assert_called_once_with("test.pdf")

    @pytest.mark.asyncio
    async def test_delete_file_returns_true_if_any_succeeds(self):
        """Test that delete_file returns True if any storage succeeds"""
        # Primary fails, fallback succeeds
        self.mock_primary.delete_file.side_effect = Exception("S3 error")
        self.mock_fallback.delete_file.return_value = True

        result = await self.manager.delete_file("test.pdf")

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_file_returns_false_if_all_fail(self):
        """Test that delete_file returns False if all storages fail"""
        # Both fail
        self.mock_primary.delete_file.return_value = False
        self.mock_fallback.delete_file.return_value = False

        result = await self.manager.delete_file("test.pdf")

        assert result is False

    def test_get_file_url_uses_primary_when_circuit_closed(self):
        """Test that get_file_url uses primary when circuit is closed"""
        self.mock_primary.get_file_url.return_value = "https://s3.url/test.pdf"

        url = self.manager.get_file_url("test.pdf")

        assert url == "https://s3.url/test.pdf"
        self.mock_primary.get_file_url.assert_called_once_with("test.pdf", 3600)

    def test_get_file_url_uses_fallback_when_circuit_open(self):
        """Test that get_file_url uses fallback when circuit is open"""
        # Open the circuit
        self.manager.circuit_open = True
        self.mock_fallback.get_file_url.return_value = "/local/path/test.pdf"

        url = self.manager.get_file_url("test.pdf")

        assert url == "/local/path/test.pdf"
        self.mock_primary.get_file_url.assert_not_called()

    def test_get_file_url_uses_fallback_on_primary_error(self):
        """Test that get_file_url uses fallback when primary fails"""
        self.mock_primary.get_file_url.side_effect = Exception("S3 error")
        self.mock_fallback.get_file_url.return_value = "/local/path/test.pdf"

        url = self.manager.get_file_url("test.pdf")

        assert url == "/local/path/test.pdf"

    def test_get_folder_name_delegates_to_primary(self):
        """Test that get_folder_name delegates to primary storage"""
        self.mock_primary.get_folder_name.return_value = "user123"

        result = self.manager.get_folder_name("user123", "agent456")

        assert result == "user123"
        self.mock_primary.get_folder_name.assert_called_once_with("user123", "agent456")

    def test_generate_storage_key_delegates_to_primary(self):
        """Test that generate_storage_key delegates to primary storage"""
        self.mock_primary.generate_storage_key.return_value = "user123/doc_abc_123.pdf"

        result = self.manager.generate_storage_key("user123", "doc.pdf", "abc123")

        assert result == "user123/doc_abc_123.pdf"
        self.mock_primary.generate_storage_key.assert_called_once_with(
            "user123", "doc.pdf", "abc123"
        )

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_no_fallback(self):
        """Test circuit breaker behavior when no fallback is available"""
        manager = StorageManager(
            primary_storage=self.mock_primary,
            fallback_storage=None,
            failure_threshold=1,
        )

        # First failure opens circuit immediately (threshold=1)
        self.mock_primary.store_file.side_effect = Exception("S3 error")

        with pytest.raises(Exception):
            await manager.store_file("/tmp/test.pdf", "test.pdf")

        assert manager.circuit_open

        # Next call should still try primary (no fallback)
        with pytest.raises(Exception):
            await manager.store_file("/tmp/test2.pdf", "test2.pdf")
