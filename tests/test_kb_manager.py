import pytest
from app.services.kb_manager import KBManager


class TestKBManager:
    def test_validate_kb_id_valid(self):
        valid_id = "kb_" + "a" * 32
        assert KBManager.validate_kb_id(valid_id) == True

    def test_validate_kb_id_invalid_format(self):
        invalid_ids = [
            "kb_123",  # too short
            "kb_" + "g" * 32,  # invalid hex
            "invalid_kb_id",  # wrong format
            "kb_" + "a" * 33,  # too long
            "kb_",  # empty uuid
            "123" + "a" * 32,  # wrong prefix
        ]
        for invalid_id in invalid_ids:
            assert KBManager.validate_kb_id(invalid_id) == False

    def test_validate_kb_id_valid_hex_chars(self):
        # Test with various valid hex characters
        valid_ids = [
            "kb_" + "0123456789abcdef" * 2,
            "kb_" + "fedcba9876543210" * 2,
            "kb_" + "a" * 32,
            "kb_" + "f" * 32,
            "kb_" + "0" * 32,
        ]
        for valid_id in valid_ids:
            assert KBManager.validate_kb_id(valid_id) == True

    def test_validate_kb_id_case_sensitive(self):
        # Should be lowercase only
        invalid_id = "kb_" + "A" * 32
        assert KBManager.validate_kb_id(invalid_id) == False
