"""
Unit tests for S3FileStorage
"""

import os
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock, ANY
from botocore.exceptions import ClientError
from app.services.storage.s3_storage import S3FileStorage


class TestS3FileStorage:
    """Test S3FileStorage functionality"""

    def setup_method(self):
        """Setup test instance with mocked S3 client"""
        with patch("boto3.client") as mock_boto3_client:
            self.mock_s3_client = Mock()
            mock_boto3_client.return_value = self.mock_s3_client
            self.storage = S3FileStorage(bucket_name="test-bucket", region="us-east-1")

    def test_init_creates_s3_client(self):
        """Test that initialization creates S3 client"""
        with patch("boto3.client") as mock_boto3_client:
            storage = S3FileStorage(bucket_name="test-bucket", region="us-west-2")
            mock_boto3_client.assert_called_once_with("s3", region_name="us-west-2")

    def test_init_handles_boto3_errors(self):
        """Test that initialization handles boto3 errors"""
        with patch("boto3.client") as mock_boto3_client:
            mock_boto3_client.side_effect = Exception("AWS credentials not found")

            with pytest.raises(Exception) as exc_info:
                S3FileStorage(bucket_name="test-bucket", region="us-east-1")

            assert "AWS credentials not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_store_file_uploads_to_s3(self):
        """Test that store_file uploads to S3"""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_file.write("Test S3 content")
            temp_file_path = temp_file.name

        try:
            # Mock put_object response
            self.mock_s3_client.put_object.return_value = {"ETag": '"abc123"'}

            # Store the file
            metadata = await self.storage.store_file(
                local_file_path=temp_file_path,
                storage_key="user123/document.pdf",
                content_type="application/pdf",
                original_filename="original.pdf",
            )

            # Verify S3 client was called correctly
            self.mock_s3_client.put_object.assert_called_once()
            call_args = self.mock_s3_client.put_object.call_args[0][0]
            assert call_args["Bucket"] == "test-bucket"
            assert call_args["Key"] == "user123/document.pdf"
            assert call_args["ContentType"] == "application/pdf"

            # Verify metadata
            assert metadata["storage_type"] == "s3"
            assert metadata["bucket"] == "test-bucket"
            assert metadata["key"] == "user123/document.pdf"
            assert metadata["folder"] == "user123"
            assert metadata["original_filename"] == "original.pdf"
            assert metadata["content_type"] == "application/pdf"
            assert metadata["size_bytes"] > 0
            assert "upload_timestamp" in metadata

        finally:
            os.unlink(temp_file_path)

    @pytest.mark.asyncio
    async def test_store_file_handles_s3_errors(self):
        """Test that store_file handles S3 errors"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_file.write("Test content")
            temp_file_path = temp_file.name

        try:
            # Mock S3 error
            self.mock_s3_client.put_object.side_effect = ClientError(
                {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}},
                "PutObject",
            )

            # Should raise the error
            with pytest.raises(ClientError):
                await self.storage.store_file(
                    local_file_path=temp_file_path, storage_key="test.txt"
                )

        finally:
            os.unlink(temp_file_path)

    @pytest.mark.asyncio
    async def test_store_file_closes_file_handle(self):
        """Test that store_file always closes file handle"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_file.write("Test content")
            temp_file_path = temp_file.name

        try:
            # Mock S3 error to trigger exception
            self.mock_s3_client.put_object.side_effect = Exception("S3 error")

            # Store should fail but still close file
            with pytest.raises(Exception):
                await self.storage.store_file(
                    local_file_path=temp_file_path, storage_key="test.txt"
                )

            # File should be closeable (not locked)
            with open(temp_file_path, "r") as f:
                assert f.read() == "Test content"

        finally:
            os.unlink(temp_file_path)

    @pytest.mark.asyncio
    async def test_delete_file_calls_s3_delete(self):
        """Test that delete_file calls S3 delete_object"""
        # Mock successful delete
        self.mock_s3_client.delete_object.return_value = {"DeleteMarker": True}

        result = await self.storage.delete_file("user123/document.pdf")

        assert result is True
        self.mock_s3_client.delete_object.assert_called_once_with(
            Bucket="test-bucket", Key="user123/document.pdf"
        )

    @pytest.mark.asyncio
    async def test_delete_file_handles_errors(self):
        """Test that delete_file handles S3 errors gracefully"""
        # Mock S3 error
        self.mock_s3_client.delete_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Key not found"}}, "DeleteObject"
        )

        result = await self.storage.delete_file("nonexistent.pdf")

        assert result is False

    def test_get_file_url_generates_presigned_url(self):
        """Test that get_file_url generates presigned URL"""
        # Mock presigned URL generation
        self.mock_s3_client.generate_presigned_url.return_value = (
            "https://test-bucket.s3.amazonaws.com/user123/doc.pdf?signature=abc123"
        )

        url = self.storage.get_file_url("user123/doc.pdf", expiration=7200)

        self.mock_s3_client.generate_presigned_url.assert_called_once_with(
            "get_object",
            Params={"Bucket": "test-bucket", "Key": "user123/doc.pdf"},
            ExpiresIn=7200,
        )
        assert url.startswith("https://")

    def test_get_file_url_handles_errors(self):
        """Test that get_file_url handles errors gracefully"""
        # Mock error
        self.mock_s3_client.generate_presigned_url.side_effect = Exception("S3 error")

        url = self.storage.get_file_url("test.pdf")

        assert url is None

    @pytest.mark.asyncio
    async def test_store_file_without_content_type(self):
        """Test storing file without explicit content type"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_file.write("Test content")
            temp_file_path = temp_file.name

        try:
            # Store without content type
            metadata = await self.storage.store_file(
                local_file_path=temp_file_path, storage_key="test.txt"
            )

            # Should not include ContentType in S3 call
            call_args = self.mock_s3_client.put_object.call_args[0][0]
            assert "ContentType" not in call_args

            # Metadata should have default content type
            assert metadata["content_type"] == "application/octet-stream"

        finally:
            os.unlink(temp_file_path)
