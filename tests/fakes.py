"""Deterministic in-process stand-ins for the inference backends.

Only the network boundary is faked. Space contract enforcement, blending,
ranking, limits and authorization all run their real code paths.
"""

import hashlib
from typing import Dict, List, Optional

from app.services import space as space_module
from app.services.space import EmbeddingSpace, SpaceSpec

FAKE_DIMENSIONS = 8
FAKE_MODEL = "fake-embedding-model"


def deterministic_vector(text: str, dimensions: int = FAKE_DIMENSIONS) -> List[float]:
    """A stable pseudo-random vector for ``text``. Never the zero vector."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [(digest[index % len(digest)] / 255.0) - 0.5 for index in range(dimensions)]


class FakeEmbeddingClient:
    """Records every batch it is asked to embed."""

    def __init__(
        self,
        dimensions: int = FAKE_DIMENSIONS,
        vectors: Optional[Dict[str, List[float]]] = None,
        error: Optional[Exception] = None,
    ):
        self.dimensions = dimensions
        self.vectors = vectors or {}
        self.error = error
        self.calls: List[List[str]] = []

    @property
    def embedded_texts(self) -> List[str]:
        return [text for call in self.calls for text in call]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.error is not None:
            raise self.error
        self.calls.append(list(texts))
        return [
            self.vectors.get(text, deterministic_vector(text, self.dimensions))
            for text in texts
        ]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def install_fake_space(
    monkeypatch,
    client: FakeEmbeddingClient,
    name: Optional[str] = None,
    model: str = FAKE_MODEL,
    normalized: bool = True,
    dimensions: Optional[int] = None,
) -> EmbeddingSpace:
    """Swap a space's backend for ``client`` for the duration of one test.

    ``dimensions`` defaults to what the client produces; passing a different
    value models a backend that drifted off its locked width.
    """
    spec = SpaceSpec(
        name=name or space_module.CHAT_SPACE_SPEC.name,
        model=model,
        dimensions=client.dimensions if dimensions is None else dimensions,
        normalized=normalized,
    )
    space = EmbeddingSpace(spec, lambda: client)
    monkeypatch.setitem(space_module._spaces, spec.name, space)
    return space
