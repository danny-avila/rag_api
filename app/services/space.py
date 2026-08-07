"""Embedding spaces served by ``POST /v1/embeddings``.

A space is a locked tuple of (model, dimensions, normalization, task prefixes).
``chat-v1`` is substitution-locked: if its backend is unavailable, or returns
vectors of the wrong width, the request fails with 503. It never silently falls
back to another model, another dimensionality, or the file-search space.

The task prefixes belong to that tuple because they change the vectors a model
produces: editing one is a space change, not a tuning knob, and stored vectors
have to be rebuilt to match.
"""

import math
import os
from dataclasses import dataclass
from numbers import Real
from typing import Callable, Dict, List, Optional

from app.config import (
    RAG_OPENAI_API_KEY,
    RAG_OPENAI_BASEURL,
    RAG_OPENAI_PROXY,
    logger,
)


INPUT_TYPE_QUERY = "query"
INPUT_TYPE_DOCUMENT = "document"


class SpaceBackendError(Exception):
    """The backend serving a space failed or answered off-contract."""


@dataclass(frozen=True)
class SpaceSpec:
    name: str
    model: str
    dimensions: int
    normalized: bool = True
    dtype: str = "float32"
    query_prefix: str = ""
    document_prefix: str = ""

    def prefix_for(self, input_type: str) -> str:
        """The task prefix this space's model expects for ``input_type``.

        Asymmetric embedding models — qwen3-embedding among them — encode a
        query and a passage differently, usually through an instruction prefix.
        Both prefixes default to empty, which is the symmetric case.
        """
        if input_type == INPUT_TYPE_QUERY:
            return self.query_prefix
        return self.document_prefix


def _finite_component(component: object) -> float:
    """One vector component, or the reason it is not one.

    ``bool`` is excluded deliberately: it is a ``Real`` in Python, and a payload
    of ``true``/``false`` is a malformed vector rather than a vector of ones and
    zeroes. NaN and infinity are rejected here because they survive
    normalization and poison every cosine score computed against them.
    """
    if isinstance(component, bool) or not isinstance(component, Real):
        raise SpaceBackendError(
            f"Backend returned a non-numeric vector component "
            f"({type(component).__name__})"
        )
    value = float(component)
    if not math.isfinite(value):
        raise SpaceBackendError("Backend returned a non-finite vector component")
    return value


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

    def _finalize(self, vectors: object, expected: int) -> List[List[float]]:
        """Check the backend's answer against the space contract.

        Everything here is a statement about a payload this process did not
        author, so every rejection is a :class:`SpaceBackendError` — including
        the shape violations that would otherwise surface as ``TypeError``. The
        route translates that single class into the documented 503; anything
        else escapes as an unhandled 500 and breaks the substitution lock.
        """
        if not isinstance(vectors, (list, tuple)):
            raise SpaceBackendError(
                f"Backend returned {type(vectors).__name__} where "
                f"{expected} vectors were expected"
            )
        if len(vectors) != expected:
            raise SpaceBackendError(
                f"Backend returned {len(vectors)} vectors for {expected} inputs"
            )
        return [self._finalize_vector(vector) for vector in vectors]

    def _finalize_vector(self, vector: object) -> List[float]:
        if not isinstance(vector, (list, tuple)):
            raise SpaceBackendError(
                f"Backend returned a {type(vector).__name__} where a vector "
                f"of {self.spec.dimensions} components was expected"
            )
        if len(vector) != self.spec.dimensions:
            raise SpaceBackendError(
                f"Backend returned {len(vector)} dimensions, "
                f"space '{self.spec.name}' is locked to {self.spec.dimensions}"
            )
        components = [_finite_component(component) for component in vector]
        return l2_normalize(components) if self.spec.normalized else components

    def _encoder(self, input_type: str) -> Callable[[List[str]], List[List[float]]]:
        """The batch encoder this space uses for ``input_type``.

        A backend that exposes a distinct batch query encoder gets to serve
        queries with it; otherwise the task prefix is what carries the
        distinction, and one batched call still serves the whole request.
        """
        client = self._client_or_raise()
        if input_type == INPUT_TYPE_QUERY:
            query_encoder = getattr(client, "embed_queries", None)
            if callable(query_encoder):
                return query_encoder
        return client.embed_documents

    def embed(
        self, texts: List[str], input_type: str = INPUT_TYPE_DOCUMENT
    ) -> List[List[float]]:
        """Embed ``texts`` the way this space encodes ``input_type``.

        Routing a query through the passage path returns a passage vector, which
        an asymmetric model scores against stored passages incorrectly. The
        caller's declared input type therefore selects the encoder and the task
        prefix rather than being recorded and ignored.
        """
        encoder = self._encoder(input_type)
        prefix = self.spec.prefix_for(input_type)
        prepared = [prefix + text for text in texts] if prefix else list(texts)
        try:
            vectors = encoder(prepared)
            return self._finalize(vectors, len(texts))
        except SpaceBackendError:
            raise
        except Exception as exc:
            # Reading the response is as much "the backend" as the call itself:
            # a malformed payload raises TypeError or ValueError from inside
            # _finalize, and the route only knows how to sanitize one class.
            raise SpaceBackendError(
                f"Embedding backend for space '{self.spec.name}' failed: {exc}"
            ) from exc

    def payload_characters(self, texts: List[str], input_type: str) -> int:
        """Characters :meth:`embed` would actually send for ``texts``.

        The task prefix is prepended to every input, so the caller's text is not
        the payload. The size limit is a provider limit and has to bind what
        leaves the process, prefix included.
        """
        prefix_length = len(self.spec.prefix_for(input_type))
        return sum(len(text) for text in texts) + prefix_length * len(texts)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed(texts, INPUT_TYPE_DOCUMENT)

    def embed_query(self, text: str) -> List[float]:
        return self.embed([text], INPUT_TYPE_QUERY)[0]


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
    query_prefix=os.getenv("RAG_CHAT_EMBEDDING_QUERY_PREFIX", ""),
    document_prefix=os.getenv("RAG_CHAT_EMBEDDING_DOCUMENT_PREFIX", ""),
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
