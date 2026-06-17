"""Tests for diagnostics helpers in document_routes."""

import pytest

from app.routes.document_routes import _is_encrypted_pdf_error


@pytest.mark.parametrize(
    "message, expected",
    [
        # Real strings captured from the pinned libraries:
        ("File has not been decrypted", True),  # pypdf FileNotDecryptedError
        (
            "Failed to load document (PDFium: Incorrect password error).",
            True,
        ),  # pypdfium2 PdfiumError on a password-protected PDF
        ("This PDF is encrypted", True),
        ("Please provide the password", True),
        # Must NOT misfire on unrelated errors:
        ("Unsupported image filter /DCTDecode", False),
        ("FileNotFoundError: no such file", False),
        ("Connection refused", False),
        ("", False),
    ],
)
def test_is_encrypted_pdf_error(message, expected):
    assert _is_encrypted_pdf_error(message) is expected
