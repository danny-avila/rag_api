"""
Tests for CVE-2025-68413 / CVE-2025-68414 path traversal fixes.

Validates that validate_file_path() correctly prevents directory traversal,
symlink escape, and other path manipulation attacks on all four protected
endpoints: /local/embed, /embed, /embed-upload, /text.

These tests are designed to catch regressions — if any test here fails,
the CVE fix is broken.
"""

import os
import pytest
from pathlib import Path

from app.routes.document_routes import validate_file_path


# ---------------------------------------------------------------------------
# Unit tests for validate_file_path()
# ---------------------------------------------------------------------------


class TestValidateFilePathTraversal:
    """Ensure directory traversal attempts are rejected."""

    @pytest.mark.parametrize(
        "malicious_path",
        [
            "../../etc/passwd",
            "../../../etc/shadow",
            "../sibling_dir/secret",
            "subdir/../../../etc/passwd",
            "subdir/./../../etc/passwd",
            "valid/../../../etc/passwd",
        ],
        ids=[
            "dotdot-etc-passwd",
            "triple-dotdot-etc-shadow",
            "dotdot-sibling",
            "subdir-then-escape",
            "dot-slash-then-escape",
            "valid-then-escape",
        ],
    )
    def test_traversal_attempts_rejected(self, tmp_path, malicious_path):
        result = validate_file_path(str(tmp_path), malicious_path)
        assert result is None, (
            f"Path traversal not blocked: validate_file_path({tmp_path!r}, {malicious_path!r}) "
            f"returned {result!r} instead of None"
        )

    @pytest.mark.parametrize(
        "absolute_path",
        [
            "/etc/passwd",
            "/etc/shadow",
            "/tmp/evil",
            "/root/.ssh/id_rsa",
        ],
        ids=[
            "abs-etc-passwd",
            "abs-etc-shadow",
            "abs-tmp-evil",
            "abs-root-ssh",
        ],
    )
    def test_absolute_path_escape_rejected(self, tmp_path, absolute_path):
        result = validate_file_path(str(tmp_path), absolute_path)
        assert result is None, (
            f"Absolute path escape not blocked: validate_file_path({tmp_path!r}, {absolute_path!r}) "
            f"returned {result!r} instead of None"
        )


class TestValidateFilePathPrefixBypass:
    """
    Regression test for the startswith() prefix-matching vulnerability.

    If base_dir = "/app/uploads", a path like "/app/uploads_evil/file"
    passes str.startswith("/app/uploads"). The fix must use path-boundary-
    aware comparison (e.g. Path.relative_to or appending os.sep).
    """

    def test_sibling_directory_with_shared_prefix(self, tmp_path):
        """CVE-2025-68413 core regression: sibling dir with same prefix."""
        base_dir = tmp_path / "uploads"
        evil_dir = tmp_path / "uploads_evil"
        base_dir.mkdir()
        evil_dir.mkdir()
        evil_file = evil_dir / "stolen.txt"
        evil_file.write_text("sensitive data")

        result = validate_file_path(str(base_dir), str(evil_file))
        assert result is None, (
            f"Prefix bypass not blocked: sibling dir 'uploads_evil' was accessible "
            f"from base 'uploads'. Got {result!r}"
        )

    def test_sibling_directory_relative_prefix_bypass(self, tmp_path):
        """Same prefix attack via relative path component."""
        base_dir = tmp_path / "uploads"
        evil_dir = tmp_path / "uploads2"
        base_dir.mkdir()
        evil_dir.mkdir()
        evil_file = evil_dir / "data.txt"
        evil_file.write_text("secret")

        # Try relative path that might resolve to uploads2/
        result = validate_file_path(str(base_dir), "../uploads2/data.txt")
        assert result is None


class TestValidateFilePathSymlinks:
    """
    Regression test for symlink traversal (os.path.abspath vs realpath).

    If base_dir contains a symlink pointing outside it, abspath won't
    detect the escape but realpath/resolve will.
    """

    def test_symlink_escape(self, tmp_path):
        """Symlink inside base_dir pointing to /tmp (outside base)."""
        base_dir = tmp_path / "uploads"
        base_dir.mkdir()
        target_dir = tmp_path / "outside"
        target_dir.mkdir()
        secret = target_dir / "secret.txt"
        secret.write_text("private data")

        link = base_dir / "escape_link"
        link.symlink_to(target_dir)

        result = validate_file_path(str(base_dir), "escape_link/secret.txt")
        assert result is None, (
            f"Symlink escape not blocked: 'escape_link' → {target_dir} "
            f"was traversable. Got {result!r}"
        )

    def test_symlink_to_parent(self, tmp_path):
        """Symlink pointing to parent directory."""
        base_dir = tmp_path / "uploads"
        base_dir.mkdir()

        link = base_dir / "parent_link"
        link.symlink_to(tmp_path)

        result = validate_file_path(str(base_dir), "parent_link/uploads/../secret")
        assert result is None


class TestValidateFilePathEdgeCases:
    """Edge cases and malformed input."""

    def test_empty_string_rejected(self, tmp_path):
        result = validate_file_path(str(tmp_path), "")
        assert result is None, "Empty string should be rejected"

    def test_whitespace_only_rejected(self, tmp_path):
        result = validate_file_path(str(tmp_path), "   ")
        assert result is None, "Whitespace-only path should be rejected"

    def test_dot_only_returns_none_or_base(self, tmp_path):
        """A single dot resolves to base_dir itself — should be rejected (not a file)."""
        result = validate_file_path(str(tmp_path), ".")
        # Accepting the base dir itself is a design choice; either None or
        # the base dir string is acceptable, but it must NOT escape.
        if result is not None:
            assert Path(result).resolve() == tmp_path.resolve()

    def test_null_byte_rejected(self, tmp_path):
        """Null bytes in filenames should be rejected or cause no harm."""
        try:
            result = validate_file_path(str(tmp_path), "file\x00.pdf")
            # If it doesn't raise, it must return None
            assert result is None, "Null byte in filename should be rejected"
        except (ValueError, TypeError):
            pass  # Raising is also acceptable

    def test_very_long_path(self, tmp_path):
        """Extremely long paths should not cause crashes."""
        long_name = "a" * 1000
        try:
            result = validate_file_path(str(tmp_path), long_name)
            # Either None or a valid path under base_dir
            if result is not None:
                assert result.startswith(str(tmp_path))
        except (OSError, ValueError):
            pass  # OS-level rejection is fine


class TestValidateFilePathValidInputs:
    """Ensure legitimate paths are accepted correctly."""

    def test_simple_filename(self, tmp_path):
        result = validate_file_path(str(tmp_path), "document.pdf")
        assert result is not None
        assert Path(result).resolve().parent == tmp_path.resolve()

    def test_filename_with_spaces(self, tmp_path):
        result = validate_file_path(str(tmp_path), "my document.pdf")
        assert result is not None
        assert str(tmp_path) in result

    def test_subdirectory_path(self, tmp_path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        result = validate_file_path(str(tmp_path), "subdir/file.txt")
        assert result is not None
        resolved = Path(result).resolve()
        assert resolved.is_relative_to(tmp_path.resolve())

    def test_returned_path_is_absolute(self, tmp_path):
        result = validate_file_path(str(tmp_path), "test.txt")
        assert result is not None
        assert os.path.isabs(result), f"Expected absolute path, got {result!r}"


# ---------------------------------------------------------------------------
# Integration tests: endpoint-level path traversal via TestClient
# ---------------------------------------------------------------------------

import datetime
import jwt
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from concurrent.futures import ThreadPoolExecutor

from main import app

client = TestClient(app)


@pytest.fixture
def auth_headers():
    jwt_secret = "testsecret"
    os.environ["JWT_SECRET"] = jwt_secret
    payload = {
        "id": "testuser",
        "exp": datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(hours=1),
    }
    token = jwt.encode(payload, jwt_secret, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def _setup_thread_pool():
    """Ensure app.state.thread_pool exists for tests."""
    if not hasattr(app.state, "thread_pool") or app.state.thread_pool is None:
        app.state.thread_pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="test-worker"
        )


class TestLocalEmbedPathTraversal:
    """CVE-2025-68413: /local/embed path traversal via document.filepath."""

    @pytest.mark.parametrize(
        "filepath",
        [
            "../../etc/passwd",
            "/etc/passwd",
            "../../../etc/shadow",
            "subdir/../../../etc/passwd",
        ],
    )
    def test_traversal_rejected(self, auth_headers, filepath):
        data = {
            "filepath": filepath,
            "filename": "evil.txt",
            "file_content_type": "text/plain",
            "file_id": "testid1",
        }
        response = client.post("/local/embed", json=data, headers=auth_headers)
        # Should get 404 (file not found) or 400 (invalid) — NOT 200
        assert response.status_code in (400, 404), (
            f"Path traversal not blocked on /local/embed with filepath={filepath!r}. "
            f"Got status {response.status_code}: {response.text}"
        )


class TestEntityIdPathTraversal:
    """Path traversal via entity_id parameter poisoning temp_base_path."""

    @pytest.mark.parametrize(
        "entity_id",
        [
            "../../etc",
            "../../../",
            "legit/../../../etc",
        ],
    )
    def test_embed_entity_id_traversal(self, auth_headers, entity_id, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        with test_file.open("rb") as f:
            response = client.post(
                "/embed",
                data={"file_id": "testid1", "entity_id": entity_id},
                files={"file": ("safe.txt", f, "text/plain")},
                headers=auth_headers,
            )
        assert response.status_code == 400, (
            f"entity_id traversal not blocked on /embed with entity_id={entity_id!r}. "
            f"Got status {response.status_code}: {response.text}"
        )

    @pytest.mark.parametrize(
        "entity_id",
        [
            "../../etc",
            "../../../",
            "legit/../../../etc",
        ],
    )
    def test_text_entity_id_traversal(self, auth_headers, entity_id, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        with test_file.open("rb") as f:
            response = client.post(
                "/text",
                data={"file_id": "testid1", "entity_id": entity_id},
                files={"file": ("safe.txt", f, "text/plain")},
                headers=auth_headers,
            )
        assert response.status_code == 400, (
            f"entity_id traversal not blocked on /text with entity_id={entity_id!r}. "
            f"Got status {response.status_code}: {response.text}"
        )


class TestEmbedPathTraversal:
    """CVE-2025-68414: /embed path traversal via filename."""

    @pytest.mark.parametrize(
        "filename",
        [
            "../../etc/passwd",
            "../../../etc/shadow",
            "/etc/passwd",
        ],
    )
    def test_traversal_rejected(self, auth_headers, filename, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        with test_file.open("rb") as f:
            response = client.post(
                "/embed",
                data={"file_id": "testid1", "entity_id": "testuser"},
                files={"file": (filename, f, "text/plain")},
                headers=auth_headers,
            )
        assert response.status_code == 400, (
            f"Path traversal not blocked on /embed with filename={filename!r}. "
            f"Got status {response.status_code}: {response.text}"
        )


class TestEmbedUploadPathTraversal:
    """CVE-2025-68414: /embed-upload path traversal via filename."""

    @pytest.mark.parametrize(
        "filename",
        [
            "../../etc/passwd",
            "../../../etc/shadow",
            "/etc/passwd",
        ],
    )
    def test_traversal_rejected(self, auth_headers, filename, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        with test_file.open("rb") as f:
            response = client.post(
                "/embed-upload",
                data={"file_id": "testid1", "entity_id": "testuser"},
                files={"uploaded_file": (filename, f, "text/plain")},
                headers=auth_headers,
            )
        assert response.status_code == 400, (
            f"Path traversal not blocked on /embed-upload with filename={filename!r}. "
            f"Got status {response.status_code}: {response.text}"
        )


class TestTextEndpointPathTraversal:
    """CVE-2025-68414: /text path traversal via filename."""

    @pytest.mark.parametrize(
        "filename",
        [
            "../../etc/passwd",
            "../../../etc/shadow",
            "/etc/passwd",
        ],
    )
    def test_traversal_rejected(self, auth_headers, filename, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        with test_file.open("rb") as f:
            response = client.post(
                "/text",
                data={"file_id": "testid1", "entity_id": "testuser"},
                files={"file": (filename, f, "text/plain")},
                headers=auth_headers,
            )
        assert response.status_code == 400, (
            f"Path traversal not blocked on /text with filename={filename!r}. "
            f"Got status {response.status_code}: {response.text}"
        )
