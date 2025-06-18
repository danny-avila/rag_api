"""
Unit tests for BaseFileStorage
"""

import pytest
from datetime import datetime
from app.services.storage.base_storage import BaseFileStorage


class TestBaseFileStorage:
    """Test BaseFileStorage functionality"""

    def setup_method(self):
        """Setup test instance"""
        self.storage = BaseFileStorage()

    def test_sanitize_path_component_removes_path_traversal(self):
        """Test that path traversal attempts are removed"""
        assert (
            self.storage.sanitize_path_component("../../../etc/passwd") == "etcpasswd"
        )
        assert (
            self.storage.sanitize_path_component("folder/../file.txt")
            == "folderfile.txt"
        )
        assert (
            self.storage.sanitize_path_component("..\\windows\\system32")
            == "windowssystem32"
        )

    def test_sanitize_path_component_removes_problematic_chars(self):
        """Test that problematic characters are replaced"""
        assert (
            self.storage.sanitize_path_component('file<>:"|?*.txt') == "file_______.txt"
        )
        assert self.storage.sanitize_path_component("my:file.pdf") == "my_file.pdf"

    def test_sanitize_path_component_limits_length(self):
        """Test that long filenames are truncated"""
        long_name = "a" * 150 + ".txt"
        result = self.storage.sanitize_path_component(long_name)
        assert len(result) == 99  # 95 chars + '.txt' (4 chars) = 99
        assert result.endswith(".txt")
        assert result.startswith("a" * 95)

    def test_generate_storage_key_format(self):
        """Test storage key generation format"""
        # Mock datetime for consistent testing
        import unittest.mock

        with unittest.mock.patch(
            "app.services.storage.base_storage.datetime"
        ) as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"

            key = self.storage.generate_storage_key(
                "user123", "document.pdf", "file-id-12345678"
            )
            assert key == "user123/document_file-id-1_20240101_120000.pdf"

    def test_generate_storage_key_sanitizes_inputs(self):
        """Test that storage key generation sanitizes inputs"""
        import unittest.mock

        with unittest.mock.patch(
            "app.services.storage.base_storage.datetime"
        ) as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"

            key = self.storage.generate_storage_key(
                "../user123", "doc:ument.pdf", "file-id-12345678"
            )
            assert key == "user123/doc_ument_file-id-1_20240101_120000.pdf"

    def test_get_folder_name_priority_agent_first(self):
        """Test that agent_id has highest priority"""
        assert self.storage.get_folder_name("user123", "agent456") == "agent456"
        assert self.storage.get_folder_name("public", "agent456") == "agent456"
        assert self.storage.get_folder_name(None, "agent456") == "agent456"

    def test_get_folder_name_user_second(self):
        """Test that user_id is used when no agent_id"""
        assert self.storage.get_folder_name("user123", None) == "user123"
        assert self.storage.get_folder_name("user123", "") == "user123"

    def test_get_folder_name_public_fallback(self):
        """Test that 'public' is used as fallback"""
        assert self.storage.get_folder_name("public", None) == "public"
        assert self.storage.get_folder_name("", None) == "public"
        assert self.storage.get_folder_name(None, None) == "public"

    def test_get_folder_name_edge_cases(self):
        """Test edge cases for folder name determination"""
        # Empty string agent_id should be treated as None
        assert self.storage.get_folder_name("user123", "") == "user123"

        # Whitespace-only should be treated as empty
        assert self.storage.get_folder_name("   ", None) == "public"

        # Valid user_id with various values
        assert self.storage.get_folder_name("0", None) == "0"  # "0" is a valid user_id
        assert self.storage.get_folder_name("user-with-dash", None) == "user-with-dash"
