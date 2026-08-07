"""fast-v1 embed-blend against real stored vectors in PostgreSQL.

Proves the two properties that only a real database can show: the stored-vector
lookup is owner-scoped in SQL, and a candidate whose vector is already stored
costs no candidate inference.
"""

import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app import auth
from app.services.vector_store.async_pg_vector import AsyncPgVector
from main import app
from tests.integration.conftest import DeterministicEmbeddings
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
    "tenant chunk": [0.7, 0.7141428428542851, 0.0],
}

# (file_id, user_id, chunk, tenant_id) — a None tenant models a chunk written
# before tenants were recorded.
CORPUS = [
    ("file-a", "user-1", "alpha chunk", None),
    ("file-a", "user-1", "beta chunk", None),
    ("file-b", "agent-7", "gamma chunk", None),
    ("file-c", "user-2", "foreign chunk", None),
    ("file-d", "user-1", "tenant chunk", "tenant-a"),
]


# The base tenant also matches chunks written before tenants were recorded.
BASE_TENANTS = ["__BASE__", None]


def digest(text_value: str) -> str:
    return hashlib.md5(text_value.encode("utf-8")).hexdigest()


def row_uuid_for(engine, store, chunk: str) -> str:
    """The row id of ``chunk`` *in this store's collection*.

    Digests collide by design — identical content hashes identically — and the
    embedding table is shared by every collection in the database, so the
    collection has to be part of the lookup.
    """
    from sqlalchemy import text

    with engine.connect() as conn:
        return conn.execute(
            text(
                "SELECT e.uuid FROM langchain_pg_embedding e "
                "JOIN langchain_pg_collection c ON c.uuid = e.collection_id "
                "WHERE c.name = :collection AND e.cmetadata->>'digest' = :digest"
            ),
            {"collection": store.collection_name, "digest": digest(chunk)},
        ).scalar()


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
    for file_id, user_id, chunk, tenant_id in CORPUS:
        metadata = {
            "file_id": file_id,
            "user_id": user_id,
            "digest": digest(chunk),
        }
        if tenant_id is not None:
            metadata["tenant_id"] = tenant_id
        pg_store.add_documents(
            [Document(page_content=chunk, metadata=metadata)], ids=[file_id]
        )
    return pg_store


@pytest.fixture()
def sibling_collection(pg_url, engine, _create_tables, monkeypatch):
    """A second PGVector collection in the same database and the same table.

    ``langchain_pg_embedding`` is shared by every collection, so this is what a
    deployment hosting more than one store actually looks like.
    """
    from app.services.vector_store.factory import get_vector_store
    from tests.conftest import ORIGINAL_PGVECTOR_POST_INIT

    monkeypatch.setattr(AsyncPgVector, "__post_init__", ORIGINAL_PGVECTOR_POST_INIT)
    embeddings = DeterministicEmbeddings(VECTORS)
    store = get_vector_store(
        connection_string=pg_url,
        embeddings=embeddings,
        collection_name=f"sibling-{uuid.uuid4().hex}",
        mode="async",
        create_extension=False,
    )
    yield store
    store._bind.dispose()


class TestCollectionIsolation:
    """One database, two collections: neither may answer for the other.

    The candidate queries used to match on id alone, so a row belonging to a
    sibling collection could be reported as existing — manufacturing a 403 for a
    candidate this store never held — or have its vector handed back and scored.
    """

    async def test_a_sibling_collections_row_is_not_reported_as_existing(
        self, seeded, sibling_collection
    ):
        shared_digest = digest("alpha chunk")
        await sibling_collection.aadd_documents(
            [
                Document(
                    page_content="alpha chunk",
                    metadata={
                        "file_id": "file-x",
                        "user_id": "stranger",
                        "digest": shared_digest,
                    },
                )
            ],
            ids=["file-x"],
        )

        existing, authorized = await sibling_collection.probe_candidate_ids(
            [digest("foreign chunk")], ["user-1"], BASE_TENANTS
        )
        # foreign chunk lives only in the seeded collection.
        assert existing == set()
        assert authorized == set()

    async def test_a_sibling_collections_vector_is_never_reused(
        self, seeded, sibling_collection
    ):
        foreign = digest("foreign chunk")
        assert (
            await sibling_collection.get_vectors_by_ids(
                [foreign], ["user-2"], BASE_TENANTS
            )
            == {}
        )
        # The same id, owner and tenant resolve fine in the collection that owns it.
        assert list(
            await seeded.get_vectors_by_ids([foreign], ["user-2"], BASE_TENANTS)
        ) == [foreign]

    async def test_each_collection_sees_only_its_own_copy(
        self, seeded, sibling_collection
    ):
        shared_digest = digest("alpha chunk")
        await sibling_collection.aadd_documents(
            [
                Document(
                    page_content="alpha chunk",
                    metadata={
                        "file_id": "file-x",
                        "user_id": "stranger",
                        "digest": shared_digest,
                    },
                )
            ],
            ids=["file-x"],
        )

        seeded_existing, seeded_authorized = await seeded.probe_candidate_ids(
            [shared_digest], ["user-1"], BASE_TENANTS
        )
        assert seeded_existing == {shared_digest}
        assert seeded_authorized == {shared_digest}

        sibling_existing, sibling_authorized = (
            await sibling_collection.probe_candidate_ids(
                [shared_digest], ["user-1"], BASE_TENANTS
            )
        )
        # The sibling holds a copy owned by someone else — existing, unauthorized.
        assert sibling_existing == {shared_digest}
        assert sibling_authorized == set()

    async def test_a_sibling_collections_row_uuid_resolves_to_nothing(
        self, seeded, sibling_collection, engine
    ):
        """The row id is globally unique, so only the collection filter stops it."""
        await sibling_collection.aadd_documents(
            [
                Document(
                    page_content="alpha chunk",
                    metadata={
                        "file_id": "file-x",
                        "user_id": "user-1",
                        "digest": digest("alpha chunk"),
                    },
                )
            ],
            ids=["file-x"],
        )
        sibling_uuid = str(row_uuid_for(engine, sibling_collection, "alpha chunk"))

        assert (
            await seeded.get_vectors_by_ids([sibling_uuid], ["user-1"], BASE_TENANTS)
            == {}
        )
        existing, authorized = await seeded.probe_candidate_ids(
            [sibling_uuid], ["user-1"], BASE_TENANTS
        )
        assert existing == set()
        assert authorized == set()

    async def test_a_collection_that_holds_nothing_authorizes_nothing(
        self, seeded, sibling_collection
    ):
        candidates = [digest(chunk) for _, _, chunk, _ in CORPUS]
        existing, authorized = await sibling_collection.probe_candidate_ids(
            candidates, ["user-1"], BASE_TENANTS
        )
        assert existing == set()
        assert authorized == set()
        assert (
            await sibling_collection.get_vectors_by_ids(
                candidates, ["user-1"], BASE_TENANTS
            )
            == {}
        )


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
        vectors = await seeded.get_vectors_by_ids(
            [digest("alpha chunk")], ["user-1"], BASE_TENANTS
        )
        assert list(vectors) == [digest("alpha chunk")]
        assert len(vectors[digest("alpha chunk")]) == 3

    async def test_row_uuid_resolves_to_the_stored_vector(self, seeded, engine):
        row_uuid = row_uuid_for(engine, seeded, "alpha chunk")
        vectors = await seeded.get_vectors_by_ids(
            [str(row_uuid)], ["user-1"], BASE_TENANTS
        )
        assert list(vectors) == [str(row_uuid)]

    async def test_foreign_owners_vector_is_never_returned(self, seeded):
        foreign = digest("foreign chunk")
        assert (
            await seeded.get_vectors_by_ids([foreign], ["user-1"], BASE_TENANTS) == {}
        )
        assert list(
            await seeded.get_vectors_by_ids([foreign], ["user-2"], BASE_TENANTS)
        ) == [foreign]

    async def test_unknown_ids_resolve_to_nothing(self, seeded):
        assert (
            await seeded.get_vectors_by_ids(["not-a-real-id"], ["user-1"], BASE_TENANTS)
            == {}
        )

    async def test_empty_owner_scope_resolves_to_nothing(self, seeded):
        assert (
            await seeded.get_vectors_by_ids([digest("alpha chunk")], [], BASE_TENANTS)
            == {}
        )

    async def test_a_non_uuid_candidate_id_does_not_break_the_query(self, seeded):
        """The uuid arm casts the column, never the caller's string."""
        result = await seeded.get_vectors_by_ids(
            ["'; DROP TABLE langchain_pg_embedding; --", digest("alpha chunk")],
            ["user-1"],
            BASE_TENANTS,
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

    def test_a_foreign_candidate_id_is_refused_before_any_egress(self, rerank_client):
        """The id resolves to a stored row, but not for this subject.

        Falling through to inference on the caller-supplied text would ship
        content the caller cannot read out to the gateway, so the request is
        refused instead of quietly proceeding.
        """
        candidates = [
            {"id": digest("foreign chunk"), "text": "gamma chunk", "base_score": 1.0}
        ]
        response = rerank(candidates)
        assert response.status_code == 403
        assert rerank_client.documents == []
        assert rerank_client.queries == []

    def test_a_cross_tenant_candidate_is_refused_before_any_egress(self, rerank_client):
        candidates = [
            {"id": digest("tenant chunk"), "text": "gamma chunk", "base_score": 1.0}
        ]
        response = rerank(candidates)
        assert response.status_code == 403
        assert rerank_client.documents == []

    def test_the_owning_tenant_reaches_its_own_chunk(self, rerank_client):
        token = strict_token(subject="user-1", tenant="tenant-a")
        candidates = [
            {"id": digest("tenant chunk"), "text": "gamma chunk", "base_score": 1.0}
        ]
        assert rerank(candidates, token=token).status_code == 200
        assert rerank_client.documents == []

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
        response = rerank(candidates)
        assert response.status_code == 403
        assert rerank_client.documents == []

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
