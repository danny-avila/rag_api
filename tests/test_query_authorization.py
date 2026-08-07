"""Regression tests for the /query and /query_multiple authorization holes.

Before the fix, ``/query`` authorized an entire result set from the metadata of
``documents[0]`` — the first *returned* hit — and ``/query_multiple`` performed
no authorization at all. Both accepted a caller-supplied ``entity_id`` as the
identity to compare against, so naming a victim's id authorized the victim's
chunks.

The store here evaluates the metadata filter the route actually builds, so a
route that stops putting the owner predicate in the query predicate fails these
tests rather than passing on a post-hoc check.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app import auth
from app.services.vector_store.async_pg_vector import AsyncPgVector
from main import app
from tests.tokens import (
    APP_SECRET,
    BASE_TENANT,
    RAG_SECRET,
    bearer,
    legacy_token,
    strict_token,
)

client = TestClient(app)

CORPUS = [
    {"file_id": "file-owned", "user_id": "user-1", "chunk": "owner chunk a"},
    {"file_id": "file-owned", "user_id": "user-1", "chunk": "owner chunk b"},
    {"file_id": "file-foreign", "user_id": "user-2", "chunk": "foreign chunk"},
    {"file_id": "file-shared", "user_id": "user-1", "chunk": "shared owner chunk"},
    {"file_id": "file-shared", "user_id": "user-2", "chunk": "shared foreign chunk"},
    {"file_id": "file-agent", "user_id": "agent-7", "chunk": "agent chunk"},
]


def _matches(metadata: Dict[str, Any], predicate: Dict[str, Any]) -> bool:
    """Minimal evaluator for the LangChain metadata-filter dialect."""
    for key, clause in predicate.items():
        if key == "$and":
            if not all(_matches(metadata, sub) for sub in clause):
                return False
            continue
        if key == "$or":
            if not any(_matches(metadata, sub) for sub in clause):
                return False
            continue
        value = metadata.get(key)
        operator, operand = next(iter(clause.items()))
        if operator == "$eq" and value != operand:
            return False
        if operator == "$in" and value not in operand:
            return False
        if operator not in ("$eq", "$in"):
            raise AssertionError(f"unexpected operator in predicate: {operator}")
    return True


@pytest.fixture
def captured_filters(monkeypatch):
    """Route the store through a filter-evaluating fake and record predicates."""
    seen: List[Dict[str, Any]] = []

    def search(embedding, k, filter=None) -> List[Tuple[Document, float]]:
        seen.append(filter)
        hits = []
        for position, row in enumerate(CORPUS):
            metadata = {"file_id": row["file_id"], "user_id": row["user_id"]}
            if filter is not None and not _matches(metadata, filter):
                continue
            hits.append(
                (Document(page_content=row["chunk"], metadata=metadata), 0.1 * position)
            )
        return hits[:k]

    async def asearch(self, embedding, k, filter=None, executor=None):
        return search(embedding, k, filter)

    monkeypatch.setattr(
        AsyncPgVector, "asimilarity_search_with_score_by_vector", asearch
    )
    monkeypatch.setattr(
        "app.services.embedding.get_cached_query_embedding",
        lambda query: [0.1, 0.2, 0.3],
    )
    monkeypatch.setattr(
        "app.routes.document_routes.get_cached_query_embedding",
        lambda query: [0.1, 0.2, 0.3],
    )
    if getattr(app.state, "thread_pool", None) is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)
    return seen


@pytest.fixture
def strict_auth(monkeypatch):
    monkeypatch.setenv("RAG_JWT_SECRET", RAG_SECRET)
    monkeypatch.setenv("JWT_SECRET", APP_SECRET)
    monkeypatch.setenv("RAG_AUTH_ACCEPT_LEGACY", "true")
    auth.reset_settings()


def owners_in(predicate: Dict[str, Any]) -> List[str]:
    for clause in predicate["$and"]:
        if "user_id" in clause:
            return clause["user_id"]["$in"]
    raise AssertionError(f"no owner predicate in {predicate}")


def test_owner_predicate_is_part_of_the_store_query(captured_filters, strict_auth):
    response = client.post(
        "/query",
        json={"query": "q", "file_id": "file-owned", "k": 10},
        headers=bearer(strict_token(subject="user-1")),
    )
    assert response.status_code == 200
    assert owners_in(captured_filters[0]) == ["user-1"]


def test_query_hides_foreign_file(captured_filters, strict_auth):
    response = client.post(
        "/query",
        json={"query": "q", "file_id": "file-foreign", "k": 10},
        headers=bearer(strict_token(subject="user-1")),
    )
    assert response.status_code == 200
    assert response.json() == []


def test_query_never_authorizes_the_whole_set_from_the_first_hit(
    captured_filters, strict_auth
):
    """file-shared holds one chunk per user; only the caller's may come back."""
    response = client.post(
        "/query",
        json={"query": "q", "file_id": "file-shared", "k": 10},
        headers=bearer(strict_token(subject="user-1")),
    )
    assert response.status_code == 200
    owners = {hit[0]["metadata"]["user_id"] for hit in response.json()}
    assert owners == {"user-1"}


def test_query_entity_id_cannot_impersonate_another_owner(
    captured_filters, strict_auth
):
    """Naming a victim as entity_id used to authorize the victim's chunks."""
    response = client.post(
        "/query",
        json={"query": "q", "file_id": "file-foreign", "k": 10, "entity_id": "user-2"},
        headers=bearer(strict_token(subject="user-1")),
    )
    assert response.status_code == 403
    assert captured_filters == []


def test_query_permitted_entity_is_scoped_to_that_entity(captured_filters, strict_auth):
    token = strict_token(subject="user-1", entities=["agent-7"])
    response = client.post(
        "/query",
        json={"query": "q", "file_id": "file-agent", "k": 10, "entity_id": "agent-7"},
        headers=bearer(token),
    )
    assert response.status_code == 200
    assert owners_in(captured_filters[0]) == ["agent-7", "user-1"]
    assert [hit[0]["metadata"]["user_id"] for hit in response.json()] == ["agent-7"]


def test_query_multiple_hides_foreign_files(captured_filters, strict_auth):
    response = client.post(
        "/query_multiple",
        json={"query": "q", "file_ids": ["file-foreign"], "k": 10},
        headers=bearer(strict_token(subject="user-1")),
    )
    assert response.status_code == 404
    assert owners_in(captured_filters[0]) == ["user-1"]


def test_query_multiple_returns_only_the_callers_chunks(captured_filters, strict_auth):
    response = client.post(
        "/query_multiple",
        json={
            "query": "q",
            "file_ids": ["file-owned", "file-foreign", "file-shared"],
            "k": 10,
        },
        headers=bearer(strict_token(subject="user-1")),
    )
    assert response.status_code == 200
    owners = {hit[0]["metadata"]["user_id"] for hit in response.json()}
    assert owners == {"user-1"}


def test_query_multiple_entity_id_is_authorized_against_the_token(
    captured_filters, strict_auth
):
    response = client.post(
        "/query_multiple",
        json={
            "query": "q",
            "file_ids": ["file-agent"],
            "k": 10,
            "entity_id": "agent-7",
        },
        headers=bearer(strict_token(subject="user-1")),
    )
    assert response.status_code == 403


def test_legacy_token_is_still_scoped_to_its_subject(captured_filters, strict_auth):
    """Legacy tokens cannot prove entity access, but they are still scoped."""
    response = client.post(
        "/query",
        json={"query": "q", "file_id": "file-shared", "k": 10},
        headers=bearer(legacy_token(user_id="user-1")),
    )
    assert response.status_code == 200
    assert owners_in(captured_filters[0]) == ["user-1"]
    owners = {hit[0]["metadata"]["user_id"] for hit in response.json()}
    assert owners == {"user-1"}


def test_legacy_token_entity_id_never_widens_beyond_two_owners(
    captured_filters, strict_auth
):
    response = client.post(
        "/query",
        json={"query": "q", "file_id": "file-shared", "k": 10, "entity_id": "agent-7"},
        headers=bearer(legacy_token(user_id="user-1")),
    )
    assert response.status_code == 200
    assert owners_in(captured_filters[0]) == ["agent-7", "user-1"]
    assert "user-2" not in owners_in(captured_filters[0])


def test_legacy_entity_id_is_still_trusted_until_the_flag_flips(
    captured_filters, strict_auth, monkeypatch
):
    """Documents the residual transition risk, and that flipping the flag ends it.

    A legacy token carries no entity list, so rag_api cannot tell a legitimate
    agent id from a victim's user id. This is the whole reason the six LibreChat
    call paths must migrate and RAG_AUTH_ACCEPT_LEGACY must go false.
    """
    body = {"query": "q", "file_id": "file-foreign", "k": 10, "entity_id": "user-2"}
    response = client.post("/query", json=body, headers=bearer(legacy_token("user-1")))
    assert response.status_code == 200
    assert owners_in(captured_filters[0]) == ["user-1", "user-2"]

    monkeypatch.setenv("RAG_AUTH_ACCEPT_LEGACY", "false")
    auth.reset_settings()
    rejected = client.post("/query", json=body, headers=bearer(legacy_token("user-1")))
    assert rejected.status_code == 401
    strict = client.post(
        "/query", json=body, headers=bearer(strict_token(subject="user-1"))
    )
    assert strict.status_code == 403


def test_tenant_claim_does_not_widen_owner_scope(captured_filters, strict_auth):
    token = strict_token(subject="user-1", tenant=BASE_TENANT)
    client.post(
        "/query",
        json={"query": "q", "file_id": "file-owned", "k": 10},
        headers=bearer(token),
    )
    assert owners_in(captured_filters[0]) == ["user-1"]
