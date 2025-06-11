"""
Unit tests for storage factory functions
"""

import pytest
from unittest.mock import patch, Mock
from app.config import StorageProvider
from app.services.storage.factory import get_file_storage, init_storage_with_fallback
from app.services.storage.local_storage import LocalFileStorage
from app.services.storage.s3_storage import S3FileStorage
from app.services.storage.storage_manager import StorageManager


class TestStorageFactory:
    """Test storage factory functions"""

    def test_get_file_storage_creates_local_storage(self):
        """Test that get_file_storage creates LocalFileStorage correctly"""
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_dir = f"{temp_dir}/custom/storage"
            storage = get_file_storage(StorageProvider.LOCAL, storage_dir=custom_dir)

            assert isinstance(storage, LocalFileStorage)
            assert storage.storage_dir == custom_dir + "/"

    def test_get_file_storage_creates_local_with_default_dir(self):
        """Test that get_file_storage uses default directory for local"""
        storage = get_file_storage(StorageProvider.LOCAL)

        assert isinstance(storage, LocalFileStorage)
        assert storage.storage_dir == "./storage/"

    @patch("boto3.client")
    def test_get_file_storage_creates_s3_storage(self, mock_boto3_client):
        """Test that get_file_storage creates S3FileStorage correctly"""
        mock_boto3_client.return_value = Mock()

        storage = get_file_storage(
            StorageProvider.S3, bucket_name="test-bucket", region="us-west-2"
        )

        assert isinstance(storage, S3FileStorage)
        assert storage.bucket_name == "test-bucket"
        assert storage.region == "us-west-2"

    def test_get_file_storage_s3_requires_params(self):
        """Test that S3 storage requires bucket_name and region"""
        with pytest.raises(ValueError) as exc_info:
            get_file_storage(StorageProvider.S3)

        assert "S3 storage requires both bucket_name and region" in str(exc_info.value)

    def test_get_file_storage_s3_missing_bucket(self):
        """Test that S3 storage fails without bucket_name"""
        with pytest.raises(ValueError) as exc_info:
            get_file_storage(StorageProvider.S3, region="us-east-1")

        assert "S3 storage requires both bucket_name and region" in str(exc_info.value)

    def test_get_file_storage_s3_missing_region(self):
        """Test that S3 storage fails without region"""
        with pytest.raises(ValueError) as exc_info:
            get_file_storage(StorageProvider.S3, bucket_name="test-bucket")

        assert "S3 storage requires both bucket_name and region" in str(exc_info.value)

    def test_get_file_storage_invalid_provider(self):
        """Test that invalid provider raises error"""
        with pytest.raises(ValueError) as exc_info:
            get_file_storage("invalid-provider")

        assert "Invalid storage provider" in str(exc_info.value)

    @patch("boto3.client")
    def test_init_storage_with_fallback_s3_primary(self, mock_boto3_client):
        """Test init_storage_with_fallback with S3 as primary"""
        import tempfile
        mock_boto3_client.return_value = Mock()

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = init_storage_with_fallback(
                s3_bucket_name="test-bucket",
                s3_region="us-east-1",
                local_storage_dir=temp_dir,
            )

            assert isinstance(manager, StorageManager)
            assert isinstance(manager.primary_storage, S3FileStorage)
            assert isinstance(manager.fallback_storage, LocalFileStorage)
            assert manager.primary_storage.bucket_name == "test-bucket"
            assert manager.fallback_storage.storage_dir == temp_dir + "/"

    def test_init_storage_with_fallback_local_only(self):
        """Test init_storage_with_fallback with only local storage"""
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = init_storage_with_fallback(local_storage_dir=temp_dir)

            assert isinstance(manager, StorageManager)
            assert isinstance(manager.primary_storage, LocalFileStorage)
            assert manager.fallback_storage is None
            assert manager.primary_storage.storage_dir == temp_dir + "/"

    def test_init_storage_with_fallback_missing_s3_region(self):
        """Test init_storage_with_fallback falls back when S3 region missing"""
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = init_storage_with_fallback(
                s3_bucket_name="test-bucket", local_storage_dir=temp_dir
            )

            # Should fall back to local storage
            assert isinstance(manager, StorageManager)
            assert isinstance(manager.primary_storage, LocalFileStorage)
            assert manager.fallback_storage is None
            assert manager.primary_storage.storage_dir == temp_dir + "/"

    @patch("boto3.client")
    def test_init_storage_with_fallback_s3_init_failure(self, mock_boto3_client):
        """Test init_storage_with_fallback handles S3 initialization failure"""
        import tempfile
        # Mock S3 client creation failure
        mock_boto3_client.side_effect = Exception("AWS credentials not found")

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = init_storage_with_fallback(
                s3_bucket_name="test-bucket",
                s3_region="us-east-1",
                local_storage_dir=temp_dir,
            )

            # Should fall back to local storage
            assert isinstance(manager, StorageManager)
            assert isinstance(manager.primary_storage, LocalFileStorage)
            assert manager.fallback_storage is None
            assert manager.primary_storage.storage_dir == temp_dir + "/"

    @patch("boto3.client")
    def test_init_storage_with_fallback_custom_circuit_breaker(self, mock_boto3_client):
        """Test init_storage_with_fallback with custom circuit breaker settings"""
        mock_boto3_client.return_value = Mock()

        manager = init_storage_with_fallback(
            s3_bucket_name="test-bucket",
            s3_region="us-east-1",
            circuit_breaker_timeout=600,
            failure_threshold=5,
        )

        assert manager.circuit_breaker_timeout == 600
        assert manager.failure_threshold == 5

    def test_init_storage_with_fallback_default_params(self):
        """Test init_storage_with_fallback with all defaults"""
        manager = init_storage_with_fallback()

        assert isinstance(manager, StorageManager)
        assert isinstance(manager.primary_storage, LocalFileStorage)
        assert manager.primary_storage.storage_dir == "./storage/"
        assert manager.fallback_storage is None
        assert manager.circuit_breaker_timeout == 300
        assert manager.failure_threshold == 3
