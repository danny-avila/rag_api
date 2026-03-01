"""
Tests for upload temp file isolation and generate_digest correctness.

Validates:
- _make_unique_temp_path produces unique paths per call (no concurrent collisions)
- _make_unique_temp_path isolates users into separate subdirectories
- _make_unique_temp_path rejects path traversal filenames
- generate_digest is consistent for all string inputs including surrogates
"""

import hashlib
import os
import pytest

from app.routes.document_routes import _make_unique_temp_path, generate_digest


class TestMakeUniqueTempPath:
    """Ensure temp file paths are unique and user-isolated."""

    def test_two_calls_same_filename_produce_different_paths(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr("app.routes.document_routes.RAG_UPLOAD_DIR", str(tmp_path))
        path_a = _make_unique_temp_path("user1", "report.pdf")
        path_b = _make_unique_temp_path("user1", "report.pdf")
        assert path_a != path_b, "Same user+filename must produce unique paths"

    def test_different_users_produce_different_directories(self, monkeypatch, tmp_path):
        monkeypatch.setattr("app.routes.document_routes.RAG_UPLOAD_DIR", str(tmp_path))
        path_a = _make_unique_temp_path("user1", "report.pdf")
        path_b = _make_unique_temp_path("user2", "report.pdf")
        assert os.path.dirname(path_a) != os.path.dirname(path_b)
        assert "/user1/" in path_a
        assert "/user2/" in path_b

    def test_preserves_file_extension(self, monkeypatch, tmp_path):
        monkeypatch.setattr("app.routes.document_routes.RAG_UPLOAD_DIR", str(tmp_path))
        path = _make_unique_temp_path("user1", "data.csv")
        assert path.endswith(".csv")

    def test_path_stays_within_upload_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr("app.routes.document_routes.RAG_UPLOAD_DIR", str(tmp_path))
        path = _make_unique_temp_path("user1", "file.txt")
        assert path.startswith(str(tmp_path))

    @pytest.mark.parametrize(
        "malicious_filename",
        [
            "../../etc/passwd",
            "../../../etc/shadow",
            "/etc/passwd",
        ],
    )
    def test_rejects_path_traversal(self, monkeypatch, tmp_path, malicious_filename):
        monkeypatch.setattr("app.routes.document_routes.RAG_UPLOAD_DIR", str(tmp_path))
        result = _make_unique_temp_path("user1", malicious_filename)
        assert result is None


class TestGenerateDigest:
    """Ensure generate_digest is correct for all inputs."""

    def test_normal_string(self):
        content = "hello world"
        expected = hashlib.md5(content.encode("utf-8")).hexdigest()
        assert generate_digest(content) == expected

    def test_empty_string(self):
        expected = hashlib.md5(b"").hexdigest()
        assert generate_digest("") == expected

    def test_unicode_content(self):
        content = "café résumé naïve"
        expected = hashlib.md5(content.encode("utf-8")).hexdigest()
        assert generate_digest(content) == expected

    def test_surrogate_characters(self):
        """Surrogate chars are stripped by encode('utf-8', 'ignore')."""
        content = "hello\ud800world"
        expected = hashlib.md5(content.encode("utf-8", "ignore")).hexdigest()
        assert generate_digest(content) == expected
        assert len(generate_digest(content)) == 32

    def test_deterministic(self):
        content = "same input"
        assert generate_digest(content) == generate_digest(content)
