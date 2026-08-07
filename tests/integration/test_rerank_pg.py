"""fast-v1 embed-blend against real stored vectors in PostgreSQL.

Proves the two properties that only a real database can show: the stored-vector
lookup is owner-scoped in SQL, and a candidate whose vector is already stored
costs no candidate inference.
"""

import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import List

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app import auth
from main import app
from tests.tokens import APP_SECRET, RAG_SECRET, bearer, strict_token

pytestmark = pytest.mark.integration

client = TestClient(app)

QUERY = "how do I rotate the signing key"

VECTORS = {
    QUERY: [1.0, 0.0, 0.0],
    "alpha chunk": [1.0, 0.0, 0.0],
    "beta chunk": [0.9, 0.4358898943540674, 0.0],
    "gamma chunk": [0.8, 0.6, 0.0],
    "foreign chunk": [0.0, 1.0, 0.0],
}

CORPUS = [
    ("file-a", "user-1", "alpha chunk"),
    ("file-a", "user-1", "beta chunk"),
    ("file-b", "agent-7", "gamma chunk"),
    ("file-c", "user-2", "foreign chunk"),
]


def digest(text_value: str) -> str:
    return hashlib.md5(text_value.encode("utf-8")).hexdigest()


class RecordingEmbedder:
    """Stands in for the inference call so candidate embeds are countable."""

    def __init__(self):
        self.queries: List[str] = []
        self.documents: List[str] = []

    def embed_query(self, text_value: str) -> List[float]:
        self.queries.append(text_value)
        return list(VECTORS.get(text_value, [0.1, 0.1, 0.1]))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.documents.extend(texts)
        return [list(VECTORS.get(text_value, [0.1, 0.1, 0.1])) for text_value in texts]


@pytest.fixture()
def seeded(pg_store):
    pg_store.embedding_function.vectors = VECTORS
    for file_id, user_id, chunk in CORPUS:
        pg_store.add_documents(
            [
                Document(
                    page_content=chunk,
                    metadata={
                        "file_id": file_id,
                        "user_id": user_id,
                        "digest": digest(chunk),
                    },
                )
            ],
            ids=[file_id],
        )
    return pg_store


@pytest.fixture()
def rerank_client(seeded, monkeypatch):
    monkeypatch.setenv("RAG_SEARCH_API_ENABLED", "true")
    monkeypatch.setenv("RAG_JWT_SECRET", RAG_SECRET)
    monkeypatch.setenv("JWT_SECRET", APP_SECRET)
    monkeypatch.setenv("RAG_RATE_LIMIT_ENABLED", "false")
    auth.reset_settings()

    embedder = RecordingEmbedder()
    monkeypatch.setattr("app.routes.search_routes.vector_store", seeded)
    monkeypatch.setattr(
        "app.services.embedding.get_cached_query_embedding", embedder.embed_query
    )
    monkeypatch.setattr("app.services.embedding.embed_texts", embedder.embed_documents)
    if getattr(app.state, "thread_pool", None) is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)
    return embedder


def rerank(candidates, token=None):
    return client.post(
        "/v1/rerank",
        json={"profile": "fast-v1", "query": QUERY, "candidates": candidates},
        headers=bearer(token or strict_token(subject="user-1")),
    )


class TestStoredVectorLookup:
    async def test_digest_resolves_to_the_stored_vector(self, seeded):
        vectors = await seeded.get_vectors_by_ids([digest("alpha chunk")], ["user-1"])
        assert list(vectors) == [digest("alpha chunk")]
        assert len(vectors[digest("alpha chunk")]) == 3

    async def test_row_uuid_resolves_to_the_stored_vector(self, seeded, engine):
        from sqlalchemy import text

        with engine.connect() as conn:
            row_uuid = conn.execute(
                text(
                    "SELECT uuid FROM langchain_pg_embedding "
                    "WHERE cmetadata->>'digest' = :digest"
                ),
                {"digest": digest("alpha chunk")},
            ).scalar()
        vectors = await seeded.get_vectors_by_ids([str(row_uuid)], ["user-1"])
        assert list(vectors) == [str(row_uuid)]

    async def test_foreign_owners_vector_is_never_returned(self, seeded):
        foreign = digest("foreign chunk")
        assert await seeded.get_vectors_by_ids([foreign], ["user-1"]) == {}
        assert list(await seeded.get_vectors_by_ids([foreign], ["user-2"])) == [foreign]

    async def test_unknown_ids_resolve_to_nothing(self, seeded):
        assert await seeded.get_vectors_by_ids(["not-a-real-id"], ["user-1"]) == {}

    async def test_empty_owner_scope_resolves_to_nothing(self, seeded):
        assert await seeded.get_vectors_by_ids([digest("alpha chunk")], []) == {}

    async def test_a_non_uuid_candidate_id_does_not_break_the_query(self, seeded):
        """The uuid arm casts the column, never the caller's string."""
        result = await seeded.get_vectors_by_ids(
            ["'; DROP TABLE langchain_pg_embedding; --", digest("alpha chunk")],
            ["user-1"],
        )
        assert list(result) == [digest("alpha chunk")]


class TestRerankOverStoredVectors:
    def test_stored_candidates_cost_no_candidate_inference(self, rerank_client):
        candidates = [
            {"id": digest("alpha chunk"), "text": "alpha chunk", "base_score": 1.0},
            {"id": digest("beta chunk"), "text": "beta chunk", "base_score": 5.0},
        ]
        response = rerank(candidates)
        assert response.status_code == 200
        assert rerank_client.queries == [QUERY]
        assert rerank_client.documents == []

    def test_only_vectorless_candidates_are_embedded(self, rerank_client):
        candidates = [
            {"id": digest("alpha chunk"), "text": "alpha chunk", "base_score": 1.0},
            {"id": "web-scrape-1", "text": "gamma chunk", "base_score": 5.0},
        ]
        assert rerank(candidates).status_code == 200
        assert rerank_client.documents == ["gamma chunk"]

    def test_a_foreign_candidate_id_never_reads_the_foreign_vector(self, rerank_client):
        """The id resolves to a stored row, but not for this subject."""
        candidates = [
            {"id": digest("foreign chunk"), "text": "gamma chunk", "base_score": 1.0}
        ]
        assert rerank(candidates).status_code == 200
        # Falls through to inference on the caller-supplied text instead.
        assert rerank_client.documents == ["gamma chunk"]

    def test_permitted_entity_vectors_are_reachable(self, rerank_client):
        token = strict_token(subject="user-1", entities=["agent-7"])
        candidates = [
            {"id": digest("gamma chunk"), "text": "gamma chunk", "base_score": 1.0}
        ]
        assert rerank(candidates, token=token).status_code == 200
        assert rerank_client.documents == []

    def test_entity_vectors_are_unreachable_without_the_claim(self, rerank_client):
        candidates = [
            {"id": digest("gamma chunk"), "text": "gamma chunk", "base_score": 1.0}
        ]
        assert rerank(candidates).status_code == 200
        assert rerank_client.documents == ["gamma chunk"]

    def test_blend_over_stored_vectors_is_not_pure_embedding_order(self, rerank_client):
        token = strict_token(subject="user-1", entities=["agent-7"])
        candidates = [
            {"id": digest("alpha chunk"), "text": "alpha chunk", "base_score": 1.0},
            {"id": digest("beta chunk"), "text": "beta chunk", "base_score": 5.0},
            {"id": digest("gamma chunk"), "text": "gamma chunk", "base_score": 10.0},
        ]
        response = rerank(candidates, token=token)
        assert response.status_code == 200
        assert rerank_client.documents == []
        order = [result["index"] for result in response.json()["results"]]
        # Cosine order is 0, 1, 2; the base scores lift gamma above beta.
        assert order == [0, 2, 1]

    def test_results_are_stable_across_repeated_calls(self, rerank_client):
        candidates = [
            {"id": digest("alpha chunk"), "text": "alpha chunk", "base_score": 1.0},
            {"id": digest("beta chunk"), "text": "beta chunk", "base_score": 5.0},
        ]
        assert rerank(candidates).json() == rerank(candidates).json()
