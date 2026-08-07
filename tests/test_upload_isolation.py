"""
Tests for upload temp file isolation and generate_digest correctness.

Validates:
- _make_unique_temp_path produces unique paths per call (no concurrent collisions)
- _make_unique_temp_path isolates users into separate subdirectories
- _make_unique_temp_path rejects path traversal filenames
- generate_digest is consistent for all string inputs including surrogates
- loader metadata cannot overwrite the identity the writer resolved from the token
"""

import hashlib
import os
from pathlib import Path
import pytest
from langchain_core.documents import Document

from app.routes.document_routes import (
    _make_unique_temp_path,
    _prepare_documents_sync,
    generate_digest,
)


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
        assert Path(path_a).parent.name == "user1"
        assert Path(path_b).parent.name == "user2"

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


class TestLoaderMetadataCannotOverrideTheWriter:
    """Document properties are attacker-controlled; the token's identity is not.

    Loaders for several formats preserve embedded document properties verbatim,
    so an uploaded file can carry a key named ``tenant_id``. Merging that after
    the resolved values let it replace them and stamp the chunks into another
    tenant, defeating the write-side tenant boundary that the read predicate
    depends on.
    """

    HOSTILE_METADATA = {
        "file_id": "victim-file",
        "user_id": "victim-user",
        "tenant_id": "victim-tenant",
        "digest": "0" * 32,
    }

    def _prepare(self, metadata):
        return _prepare_documents_sync(
            [Document(page_content="chunk body", metadata=dict(metadata))],
            file_id="real-file",
            user_id="real-user",
            clean_content=False,
            tenant_id="real-tenant",
        )

    @pytest.mark.parametrize("key", sorted(HOSTILE_METADATA))
    def test_a_single_hostile_key_is_overwritten(self, key):
        prepared = self._prepare({key: self.HOSTILE_METADATA[key]})
        assert prepared
        for document in prepared:
            assert document.metadata[key] != self.HOSTILE_METADATA[key]

    def test_every_trusted_key_wins_at_once(self):
        prepared = self._prepare(self.HOSTILE_METADATA)
        assert prepared
        for document in prepared:
            assert document.metadata["file_id"] == "real-file"
            assert document.metadata["user_id"] == "real-user"
            assert document.metadata["tenant_id"] == "real-tenant"
            assert document.metadata["digest"] == generate_digest(document.page_content)

    def test_the_write_lands_in_the_writers_tenant_not_the_crafted_one(self):
        prepared = self._prepare({"tenant_id": "victim-tenant"})
        assert {document.metadata["tenant_id"] for document in prepared} == {
            "real-tenant"
        }

    def test_harmless_loader_metadata_is_still_preserved(self):
        prepared = self._prepare({"source": "report.pdf", "page": 3})
        for document in prepared:
            assert document.metadata["source"] == "report.pdf"
            assert document.metadata["page"] == 3
            assert document.metadata["user_id"] == "real-user"

    def test_documents_without_metadata_are_unaffected(self):
        prepared = _prepare_documents_sync(
            [Document(page_content="chunk body")],
            file_id="real-file",
            user_id="real-user",
            clean_content=False,
            tenant_id="real-tenant",
        )
        assert prepared[0].metadata["file_id"] == "real-file"
        assert prepared[0].metadata["tenant_id"] == "real-tenant"
