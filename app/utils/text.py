"""Text normalization and non-reversible fingerprints.

The search stack bans raw search text from rag_api logs, so every place that
would otherwise log a query or a candidate body logs a fingerprint from here
instead.
"""

import re
import hashlib
import unicodedata

_WHITESPACE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    """NFKC-normalize, collapse whitespace runs, and trim.

    Mirrors the projector-side normalization so a content hash computed here
    matches the embedding-input hash the caller stores. The transform is
    idempotent, so already-normalized input is passed through unchanged.
    """
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", value)).strip()


def content_hash(value: str) -> str:
    """SHA-256 hex digest of ``value`` (expected to be normalized already)."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def fingerprint(value: str) -> str:
    """Short non-reversible handle safe to put in logs."""
    return content_hash(value)[:12]
