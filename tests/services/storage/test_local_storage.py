"""
Unit tests for LocalFileStorage
"""

import os
import tempfile
import shutil
import pytest
from app.services.storage.local_storage import LocalFileStorage


class TestLocalFileStorage:
    """Test LocalFileStorage functionality"""

    def setup_method(self):
        """Setup test instance with temporary directory"""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = LocalFileStorage(storage_dir=self.temp_dir)

    def teardown_method(self):
        """Cleanup temporary directory"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_init_creates_directory(self):
        """Test that initialization creates storage directory"""
        test_dir = os.path.join(self.temp_dir, "test_storage")
        LocalFileStorage(storage_dir=test_dir)
        assert os.path.exists(test_dir)

    @pytest.mark.asyncio
    async def test_store_file_creates_structure(self):
        """Test that store_file creates proper directory structure"""
        # Create a temporary file to store
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_file.write("Test content")
            temp_file_path = temp_file.name

        try:
            # Store the file
            metadata = await self.storage.store_file(
                local_file_path=temp_file_path,
                storage_key="user123/test_doc.txt",
                content_type="text/plain",
                original_filename="original.txt",
            )

            # Verify file was stored
            stored_path = os.path.join(self.temp_dir, "user123", "test_doc.txt")
            assert os.path.exists(stored_path)

            # Verify content
            with open(stored_path, "r") as f:
                assert f.read() == "Test content"

            # Verify metadata
            assert metadata["storage_type"] == "local"
            assert metadata["key"] == "user123/test_doc.txt"
            assert metadata["folder"] == "user123"
            assert metadata["original_filename"] == "original.txt"
            assert metadata["content_type"] == "text/plain"
            assert metadata["size_bytes"] > 0
            assert "upload_timestamp" in metadata

        finally:
            os.unlink(temp_file_path)

    @pytest.mark.asyncio
    async def test_store_file_preserves_metadata(self):
        """Test that store_file preserves file metadata"""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_file.write("Test content")
            temp_file_path = temp_file.name

        try:
            # Get original file stats
            original_stats = os.stat(temp_file_path)

            # Store the file
            await self.storage.store_file(
                local_file_path=temp_file_path, storage_key="test/file.txt"
            )

            # Verify metadata was preserved
            stored_path = os.path.join(self.temp_dir, "test", "file.txt")
            stored_stats = os.stat(stored_path)

            # shutil.copy2 preserves mtime
            assert stored_stats.st_mtime == original_stats.st_mtime

        finally:
            os.unlink(temp_file_path)

    @pytest.mark.asyncio
    async def test_delete_file_removes_file(self):
        """Test that delete_file removes the file"""
        # Create a file directly
        file_path = os.path.join(self.temp_dir, "user123", "test.txt")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write("Test content")

        # Delete the file
        result = await self.storage.delete_file("user123/test.txt")

        assert result is True
        assert not os.path.exists(file_path)

    @pytest.mark.asyncio
    async def test_delete_file_nonexistent(self):
        """Test that delete_file handles nonexistent files gracefully"""
        result = await self.storage.delete_file("nonexistent/file.txt")
        assert result is True  # Should return True even if file doesn't exist

    @pytest.mark.asyncio
    async def test_delete_file_handles_errors(self):
        """Test that delete_file handles errors gracefully"""
        # Try to delete with problematic path (attempting to delete directory itself)
        result = await self.storage.delete_file(".")
        assert result is False  # Should return False on error (can't delete directory)

    def test_get_file_url_returns_path(self):
        """Test that get_file_url returns local path"""
        url = self.storage.get_file_url("user123/document.pdf")
        expected_path = os.path.join(self.temp_dir, "user123/document.pdf")
        assert url == expected_path

    def test_get_file_url_ignores_expiration(self):
        """Test that expiration parameter is ignored for local storage"""
        url1 = self.storage.get_file_url("test.pdf", expiration=3600)
        url2 = self.storage.get_file_url("test.pdf", expiration=7200)
        assert url1 == url2

    @pytest.mark.asyncio
    async def test_store_file_with_nested_directories(self):
        """Test storing files in deeply nested directories"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_file.write("Nested content")
            temp_file_path = temp_file.name

        try:
            # Store in nested directory
            metadata = await self.storage.store_file(
                local_file_path=temp_file_path,
                storage_key="level1/level2/level3/file.txt",
            )

            # Verify file exists
            nested_path = os.path.join(
                self.temp_dir, "level1", "level2", "level3", "file.txt"
            )
            assert os.path.exists(nested_path)
            assert metadata["folder"] == "level1"

        finally:
            os.unlink(temp_file_path)
