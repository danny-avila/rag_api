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
    # Same owner, different tenant, and a pre-tenant chunk with no tenant_id.
    {
        "file_id": "file-owned",
        "user_id": "user-1",
        "tenant_id": "tenant-b",
        "chunk": "other tenant chunk",
    },
    {"file_id": "file-legacy", "user_id": "user-1", "chunk": "untagged legacy chunk"},
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
            if "tenant_id" in row:
                metadata["tenant_id"] = row["tenant_id"]
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


def _clause(predicate: Dict[str, Any], key: str) -> List[Any]:
    for clause in predicate["$and"]:
        if key in clause:
            return clause[key]["$in"]
    raise AssertionError(f"no {key} predicate in {predicate}")


def owners_in(predicate: Dict[str, Any]) -> List[str]:
    return _clause(predicate, "user_id")


def tenants_in(predicate: Dict[str, Any]) -> List[Any]:
    return _clause(predicate, "tenant_id")


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


class TestTenantScope:
    """Tenant is part of the store predicate, alongside owner and file id."""

    def _query(self, file_id, tenant, subject="user-1"):
        return client.post(
            "/query",
            json={"query": "q", "file_id": file_id, "k": 10},
            headers=bearer(strict_token(subject=subject, tenant=tenant)),
        )

    def test_the_predicate_carries_tenant_owner_and_file(
        self, captured_filters, strict_auth
    ):
        self._query("file-owned", "tenant-a")
        predicate = captured_filters[0]
        assert tenants_in(predicate) == ["tenant-a"]
        assert owners_in(predicate) == ["user-1"]
        assert {"file_id": {"$eq": "file-owned"}} in predicate["$and"]

    def test_the_same_owner_in_another_tenant_is_invisible(
        self, captured_filters, strict_auth
    ):
        response = self._query("file-owned", "tenant-a")
        assert response.status_code == 200
        # Only the tenant-b copy carries tenant_id; everything else is untagged.
        assert response.json() == []

    def test_base_tenant_absorbs_chunks_written_before_tenants_existed(
        self, captured_filters, strict_auth
    ):
        response = self._query("file-legacy", BASE_TENANT)
        assert response.status_code == 200
        assert tenants_in(captured_filters[0]) == [BASE_TENANT, None]
        assert len(response.json()) == 1

    def test_a_real_tenant_never_absorbs_untagged_chunks(
        self, captured_filters, strict_auth
    ):
        response = self._query("file-legacy", "tenant-a")
        assert response.status_code == 200
        assert tenants_in(captured_filters[0]) == ["tenant-a"]
        assert response.json() == []

    def test_a_foreign_tenants_chunk_is_invisible(self, captured_filters, strict_auth):
        response = self._query("file-owned", "tenant-b", subject="user-2")
        assert response.status_code == 200
        assert response.json() == []

    def test_query_multiple_is_tenant_scoped_too(self, captured_filters, strict_auth):
        response = client.post(
            "/query_multiple",
            json={"query": "q", "file_ids": ["file-owned", "file-legacy"], "k": 10},
            headers=bearer(strict_token(subject="user-1", tenant="tenant-a")),
        )
        assert response.status_code == 404
        assert tenants_in(captured_filters[0]) == ["tenant-a"]

    def test_system_tenant_is_refused_before_any_store_query(
        self, captured_filters, strict_auth
    ):
        response = self._query("file-owned", "__SYSTEM__")
        assert response.status_code == 403
        assert captured_filters == []


class TestDocumentedAtlasIndex:
    """The documented Atlas index has to cover what the predicate pre-filters on.

    On Atlas the scope clauses become ``$vectorSearch`` pre-filters, and Atlas
    rejects a pre-filter that references a path the index does not declare as a
    filter field. An index carrying only ``file_id`` therefore makes /query and
    /query_multiple fail outright, so the README's definition is not decoration
    — it is part of the deployment contract, and it drifts silently.
    """

    def _documented_filter_paths(self) -> set:
        import json
        import re
        from pathlib import Path

        readme = Path(__file__).resolve().parents[1] / "README.md"
        blocks = re.findall(
            r"```json\n(.*?)\n```", readme.read_text(encoding="utf-8"), re.DOTALL
        )
        for block in blocks:
            definition = json.loads(block)
            fields = definition.get("fields")
            if not isinstance(fields, list):
                continue
            if not any(field.get("type") == "vector" for field in fields):
                continue
            return {field["path"] for field in fields if field.get("type") == "filter"}
        raise AssertionError("no Atlas vector index definition found in README.md")

    def test_every_scoped_path_is_declared_as_a_filter_field(self):
        from app.scope import ScopeFilter

        scope = ScopeFilter(tenant="tenant-a", owners=("user-1",))
        referenced = {key for clause in scope.scope_clauses() for key in clause}
        documented = self._documented_filter_paths()
        assert referenced <= documented, (
            f"README Atlas index is missing filter fields for "
            f"{sorted(referenced - documented)}"
        )

    def test_the_file_predicate_is_declared_too(self):
        from app.scope import file_clause

        assert set(file_clause("file-1")) <= self._documented_filter_paths()
