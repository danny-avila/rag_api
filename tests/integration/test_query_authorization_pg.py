"""Cross-user isolation of /query and /query_multiple, executed by real PostgreSQL.

The unit suite proves the owner predicate is built and honoured by a fake
store. These tests prove PostgreSQL itself enforces it: the documents are
written through the real pgvector write path, and the filter is translated to
SQL and executed by the database.
"""

import hashlib
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app import auth
from main import app
from tests.tokens import APP_SECRET, RAG_SECRET, bearer, legacy_token, strict_token

pytestmark = pytest.mark.integration

client = TestClient(app)

CORPUS = [
    ("file-owned", "user-1", "owner chunk one"),
    ("file-owned", "user-1", "owner chunk two"),
    ("file-foreign", "user-2", "foreign chunk"),
    ("file-shared", "user-1", "shared chunk owned by user one"),
    ("file-shared", "user-2", "shared chunk owned by user two"),
    ("file-agent", "agent-7", "agent knowledge chunk"),
]


def digest(text_value: str) -> str:
    return hashlib.md5(text_value.encode("utf-8")).hexdigest()


@pytest.fixture()
def seeded(pg_store, monkeypatch):
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

    monkeypatch.setenv("RAG_JWT_SECRET", RAG_SECRET)
    monkeypatch.setenv("JWT_SECRET", APP_SECRET)
    monkeypatch.setenv("RAG_AUTH_ACCEPT_LEGACY", "true")
    auth.reset_settings()

    monkeypatch.setattr("app.routes.document_routes.vector_store", pg_store)
    monkeypatch.setattr(
        "app.routes.document_routes.get_cached_query_embedding",
        pg_store.embedding_function.embed_query,
    )
    if getattr(app.state, "thread_pool", None) is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)
    return pg_store


def query(file_id, subject="user-1", entity_id=None, token=None, k=20):
    body = {"query": "chunk", "file_id": file_id, "k": k}
    if entity_id:
        body["entity_id"] = entity_id
    return client.post(
        "/query", json=body, headers=bearer(token or strict_token(subject=subject))
    )


def owners_of(response):
    return {hit[0]["metadata"]["user_id"] for hit in response.json()}


def test_foreign_file_returns_nothing(seeded):
    response = query("file-foreign", subject="user-1")
    assert response.status_code == 200
    assert response.json() == []


def test_shared_file_returns_only_the_callers_chunks(seeded):
    response = query("file-shared", subject="user-1")
    assert response.status_code == 200
    assert owners_of(response) == {"user-1"}


def test_each_user_sees_only_their_own_side_of_a_shared_file(seeded):
    assert owners_of(query("file-shared", subject="user-1")) == {"user-1"}
    assert owners_of(query("file-shared", subject="user-2")) == {"user-2"}


def test_entity_impersonation_is_refused(seeded):
    response = query("file-foreign", subject="user-1", entity_id="user-2")
    assert response.status_code == 403


def test_permitted_entity_reaches_only_that_entitys_chunks(seeded):
    token = strict_token(subject="user-1", entities=["agent-7"])
    response = query("file-agent", token=token, entity_id="agent-7")
    assert response.status_code == 200
    assert owners_of(response) == {"agent-7"}


def test_unpermitted_entity_cannot_reach_agent_chunks(seeded):
    response = query("file-agent", subject="user-1")
    assert response.status_code == 200
    assert response.json() == []


def test_legacy_token_is_scoped_to_its_subject(seeded):
    response = query("file-shared", token=legacy_token(user_id="user-1"))
    assert response.status_code == 200
    assert owners_of(response) == {"user-1"}


def test_query_multiple_filters_foreign_files_in_sql(seeded):
    response = client.post(
        "/query_multiple",
        json={
            "query": "chunk",
            "file_ids": ["file-owned", "file-foreign", "file-shared", "file-agent"],
            "k": 20,
        },
        headers=bearer(strict_token(subject="user-1")),
    )
    assert response.status_code == 200
    assert owners_of(response) == {"user-1"}
    files = {hit[0]["metadata"]["file_id"] for hit in response.json()}
    assert files == {"file-owned", "file-shared"}


def test_query_multiple_over_only_foreign_files_returns_nothing(seeded):
    response = client.post(
        "/query_multiple",
        json={"query": "chunk", "file_ids": ["file-foreign", "file-agent"], "k": 20},
        headers=bearer(strict_token(subject="user-1")),
    )
    assert response.status_code == 404
