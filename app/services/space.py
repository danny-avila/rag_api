"""Embedding spaces served by ``POST /v1/embeddings``.

A space is a locked tuple of (model, dimensions, normalization). ``chat-v1`` is
substitution-locked: if its backend is unavailable, or returns vectors of the
wrong width, the request fails with 503. It never silently falls back to
another model, another dimensionality, or the file-search space.
"""

import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from app.config import (
    RAG_OPENAI_API_KEY,
    RAG_OPENAI_BASEURL,
    RAG_OPENAI_PROXY,
    logger,
)


class SpaceBackendError(Exception):
    """The backend serving a space failed or answered off-contract."""


@dataclass(frozen=True)
class SpaceSpec:
    name: str
    model: str
    dimensions: int
    normalized: bool = True
    dtype: str = "float32"


def l2_normalize(vector: List[float]) -> List[float]:
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        raise SpaceBackendError(
            "Backend returned a zero vector, which cannot be normalized"
        )
    return [component / norm for component in vector]


class EmbeddingSpace:
    """Lazily-built client for one space, with contract enforcement on egress."""

    def __init__(self, spec: SpaceSpec, client_factory: Callable[[], object]):
        self.spec = spec
        self._client_factory = client_factory
        self._client: Optional[object] = None

    def _client_or_raise(self):
        if self._client is None:
            try:
                self._client = self._client_factory()
            except Exception as exc:
                raise SpaceBackendError(
                    f"Failed to initialize backend for space '{self.spec.name}': {exc}"
                ) from exc
        return self._client

    def _finalize(self, vectors: List[List[float]], expected: int) -> List[List[float]]:
        if len(vectors) != expected:
            raise SpaceBackendError(
                f"Backend returned {len(vectors)} vectors for {expected} inputs"
            )
        finalized = []
        for vector in vectors:
            if len(vector) != self.spec.dimensions:
                raise SpaceBackendError(
                    f"Backend returned {len(vector)} dimensions, "
                    f"space '{self.spec.name}' is locked to {self.spec.dimensions}"
                )
            finalized.append(
                l2_normalize(vector) if self.spec.normalized else list(vector)
            )
        return finalized

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        client = self._client_or_raise()
        try:
            vectors = client.embed_documents(texts)
        except SpaceBackendError:
            raise
        except Exception as exc:
            raise SpaceBackendError(
                f"Embedding backend for space '{self.spec.name}' failed: {exc}"
            ) from exc
        return self._finalize(vectors, len(texts))

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _build_openai_client(spec: SpaceSpec):
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=spec.model,
        api_key=os.getenv("RAG_CHAT_EMBEDDING_API_KEY") or RAG_OPENAI_API_KEY,
        openai_api_base=os.getenv("RAG_CHAT_EMBEDDING_BASEURL") or RAG_OPENAI_BASEURL,
        openai_proxy=RAG_OPENAI_PROXY,
        dimensions=spec.dimensions,
        # The gateway serves models tiktoken has never heard of; client-side
        # context measurement would fail before the request is even sent.
        check_embedding_ctx_length=False,
    )


CHAT_SPACE_NAME = os.getenv("RAG_EMBEDDING_SPACE", "chat-v1").strip() or "chat-v1"

CHAT_SPACE_SPEC = SpaceSpec(
    name=CHAT_SPACE_NAME,
    model=os.getenv("RAG_CHAT_EMBEDDING_MODEL", "qwen3-embedding-8b").strip(),
    dimensions=_int_env("RAG_CHAT_EMBEDDING_DIMENSIONS", 1024),
    normalized=True,
    dtype="float32",
)

_spaces: Dict[str, EmbeddingSpace] = {
    CHAT_SPACE_SPEC.name: EmbeddingSpace(
        CHAT_SPACE_SPEC, lambda: _build_openai_client(CHAT_SPACE_SPEC)
    )
}


def get_space(name: str) -> Optional[EmbeddingSpace]:
    return _spaces.get(name)


def known_spaces() -> List[str]:
    return sorted(_spaces)


def register_space(space: EmbeddingSpace) -> None:
    """Register or replace a space. Used by tests to inject a fake backend."""
    logger.debug("Registering embedding space %s", space.spec.name)
    _spaces[space.spec.name] = space
