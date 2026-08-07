"""Optional live check of the chat-v1 space against the inference gateway.

Skipped unless ``RAG_GATEWAY_TEST_BASEURL`` and ``RAG_GATEWAY_TEST_API_KEY``
are set, because the gateway is reached over a tunnel that can disappear at any
time. Everything else in the suite runs against fakes.

    RAG_GATEWAY_TEST_BASEURL=http://localhost:18080/v1 \
    RAG_GATEWAY_TEST_API_KEY=... \
    pytest tests/integration/test_gateway_embeddings.py -m integration
"""

import math
import os

import pytest

from app.services.space import EmbeddingSpace, SpaceSpec, _build_openai_client

BASE_URL = os.getenv("RAG_GATEWAY_TEST_BASEURL")
API_KEY = os.getenv("RAG_GATEWAY_TEST_API_KEY")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (BASE_URL and API_KEY),
        reason="RAG_GATEWAY_TEST_BASEURL / RAG_GATEWAY_TEST_API_KEY are not set",
    ),
]


@pytest.fixture()
def chat_space(monkeypatch):
    monkeypatch.setenv("RAG_CHAT_EMBEDDING_BASEURL", BASE_URL)
    monkeypatch.setenv("RAG_CHAT_EMBEDDING_API_KEY", API_KEY)
    spec = SpaceSpec(
        name="chat-v1",
        model=os.getenv("RAG_GATEWAY_TEST_MODEL", "qwen3-embedding-8b"),
        dimensions=1024,
        normalized=True,
    )
    return EmbeddingSpace(spec, lambda: _build_openai_client(spec))


def test_gateway_serves_the_locked_dimensionality(chat_space):
    vectors = chat_space.embed_documents(["hello world", "a second input"])
    assert len(vectors) == 2
    assert all(len(vector) == 1024 for vector in vectors)


def test_vectors_leave_the_service_l2_normalized(chat_space):
    """The gateway returns unnormalized vectors; the space normalizes them."""
    vector = chat_space.embed_query("normalize me")
    assert math.isclose(
        math.sqrt(sum(component**2 for component in vector)), 1.0, rel_tol=1e-6
    )


def test_the_same_text_embeds_to_the_same_vector(chat_space):
    first = chat_space.embed_query("stability check")
    second = chat_space.embed_query("stability check")
    assert first == second
