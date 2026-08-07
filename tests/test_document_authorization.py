"""Regression tests for the unscoped file-addressed document routes.

``GET /ids``, ``GET /documents``, ``GET /documents/{id}/context`` and
``DELETE /documents`` addressed the store by caller-supplied file id alone. A
file id is not an authorization: anyone who learns one — or simply lists them
from ``/ids`` — could read another owner's document text or delete their chunks.
The strict tokens this release introduces made that reachable from any service
credential at all.

Those routes now carry their own scope, ``rag:documents``, so a token minted to
delete a file also has to be minted for the document plane — the inference
scopes buy nothing here, and this scope buys no inference. The two planes are
tested against each other in :class:`TestThePlanesAreSeparable`.

The store below evaluates the owner and tenant predicate the route actually
passes, and answers unscoped when it is passed none. A route that stops putting
scope in the query therefore fails these tests rather than passing on a check
performed somewhere else.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app import auth
from app.services import space as space_module
from app.services.vector_store.async_pg_vector import AsyncPgVector
from main import app
from tests.fakes import FakeEmbeddingClient, install_fake_space
from tests.tokens import APP_SECRET, RAG_SECRET, bearer, legacy_token, strict_token

client = TestClient(app)

CORPUS = [
    {"file_id": "file-owned", "user_id": "user-1", "tenant_id": "__BASE__"},
    {"file_id": "file-foreign", "user_id": "user-2", "tenant_id": "__BASE__"},
    {"file_id": "file-agent", "user_id": "agent-7", "tenant_id": "__BASE__"},
    {"file_id": "file-other-tenant", "user_id": "user-1", "tenant_id": "tenant-b"},
    # One file id, two owners — a file id is chosen by whoever uploads.
    {"file_id": "file-collided", "user_id": "user-1", "tenant_id": "__BASE__"},
    {"file_id": "file-collided", "user_id": "user-2", "tenant_id": "__BASE__"},
    # Written before tenants were recorded: no tenant_id at all.
    {"file_id": "file-untagged", "user_id": "user-1"},
]


def _in_scope(
    row: Dict[str, str],
    owners: Optional[Sequence[str]],
    tenants: Optional[Sequence[Optional[str]]],
) -> bool:
    """Whether ``row`` satisfies the scope the route passed.

    ``None`` for either argument models the store call the routes made before
    the fix — no owner predicate, no tenant predicate, every row visible.
    """
    if owners is None and tenants is None:
        return True
    return row["user_id"] in owners and row.get("tenant_id") in tenants


@pytest.fixture
def rows(monkeypatch) -> List[Dict[str, str]]:
    live = [dict(row) for row in CORPUS]

    def visible(owners, tenants, ids=None):
        return [
            row
            for row in live
            if (ids is None or row["file_id"] in ids)
            and _in_scope(row, owners, tenants)
        ]

    async def get_all_ids(self, owners=None, tenants=None, executor=None):
        return sorted({row["file_id"] for row in visible(owners, tenants)})

    async def get_filtered_ids(self, ids, owners=None, tenants=None, executor=None):
        return sorted({row["file_id"] for row in visible(owners, tenants, ids)})

    async def get_documents_by_ids(self, ids, owners=None, tenants=None, executor=None):
        return [
            Document(page_content=f"{row['file_id']} content", metadata=dict(row))
            for row in visible(owners, tenants, ids)
        ]

    async def delete_scoped(self, ids, owners=None, tenants=None, executor=None):
        doomed = visible(owners, tenants, ids)
        live[:] = [row for row in live if row not in doomed]

    async def delete(self, ids=None, collection_only=False, executor=None):
        """The unscoped delete the route reached for before the fix."""
        live[:] = [row for row in live if row["file_id"] not in (ids or [])]

    monkeypatch.setattr(AsyncPgVector, "get_all_ids", get_all_ids)
    monkeypatch.setattr(AsyncPgVector, "get_filtered_ids", get_filtered_ids)
    monkeypatch.setattr(AsyncPgVector, "get_documents_by_ids", get_documents_by_ids)
    monkeypatch.setattr(AsyncPgVector, "delete_scoped", delete_scoped, raising=False)
    monkeypatch.setattr(AsyncPgVector, "delete", delete)
    if getattr(app.state, "thread_pool", None) is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)
    return live


@pytest.fixture
def strict_auth(monkeypatch):
    monkeypatch.setenv("RAG_JWT_SECRET", RAG_SECRET)
    monkeypatch.setenv("JWT_SECRET", APP_SECRET)
    monkeypatch.setenv("RAG_AUTH_ACCEPT_LEGACY", "true")
    auth.reset_settings()


def documents_token(subject: str = "attacker", **kwargs) -> str:
    """The credential these routes now require: strict, ``rag:documents`` alone."""
    return strict_token(subject=subject, scopes=("rag:documents",), **kwargs)


def file_ids(rows: List[Dict[str, str]]) -> List[str]:
    return [row["file_id"] for row in rows]


def get_documents(ids, token, entity_id=None):
    params = {"ids": ids}
    if entity_id is not None:
        params["entity_id"] = entity_id
    return client.get("/documents", params=params, headers=bearer(token))


def get_context(file_id, token, entity_id=None):
    url = f"/documents/{file_id}/context"
    if entity_id is not None:
        url = f"{url}?entity_id={entity_id}"
    return client.get(url, headers=bearer(token))


def delete_documents(ids, token, entity_id=None):
    url = "/documents"
    if entity_id is not None:
        url = f"{url}?entity_id={entity_id}"
    return client.request("DELETE", url, json=ids, headers=bearer(token))


class TestADocumentTokenCannotReadByFileIdAlone:
    """The credential in the finding: strict, RAG-signed, scoped to documents.

    Holding the document scope is permission to address this plane, never
    permission to address another owner's rows inside it.
    """

    def test_get_documents_refuses_a_foreign_file(self, rows, strict_auth):
        response = get_documents(["file-foreign"], documents_token())
        assert response.status_code == 404
        assert "file-foreign content" not in response.text

    def test_get_context_refuses_a_foreign_file(self, rows, strict_auth):
        response = get_context("file-foreign", documents_token())
        assert response.status_code == 404
        assert "file-foreign content" not in response.text

    def test_delete_refuses_a_foreign_file(self, rows, strict_auth):
        response = delete_documents(["file-foreign"], documents_token())
        assert response.status_code == 404
        assert "file-foreign" in file_ids(rows)

    def test_ids_does_not_enumerate_the_deployment(self, rows, strict_auth):
        response = client.get("/ids", headers=bearer(documents_token()))
        assert response.status_code == 200
        assert response.json() == []

    def test_a_mixed_request_leaks_nothing(self, rows, strict_auth):
        """One readable id alongside a foreign one must not carry it along."""
        token = documents_token(subject="user-1")
        response = get_documents(["file-owned", "file-foreign"], token)
        assert response.status_code == 404
        assert "file-foreign content" not in response.text


class TestLegacyTokensAreScopedToo:
    """The grandfather covers scopes and entities, never other owners' files."""

    def test_get_documents_refuses_another_users_file(self, rows, strict_auth):
        token = legacy_token(user_id="user-1")
        assert get_documents(["file-foreign"], token).status_code == 404

    def test_get_context_refuses_another_users_file(self, rows, strict_auth):
        token = legacy_token(user_id="user-1")
        assert get_context("file-foreign", token).status_code == 404

    def test_delete_refuses_another_users_file(self, rows, strict_auth):
        token = legacy_token(user_id="user-1")
        assert delete_documents(["file-foreign"], token).status_code == 404
        assert "file-foreign" in file_ids(rows)


class TestTheOwnersOwnFilesStillWork:
    def test_get_documents_returns_the_callers_file(self, rows, strict_auth):
        response = get_documents(["file-owned"], documents_token(subject="user-1"))
        assert response.status_code == 200
        assert response.json()[0]["page_content"] == "file-owned content"

    def test_get_context_returns_the_callers_file(self, rows, strict_auth):
        response = get_context("file-owned", documents_token(subject="user-1"))
        assert response.status_code == 200
        assert "file-owned content" in response.text

    def test_delete_removes_the_callers_file(self, rows, strict_auth):
        response = delete_documents(["file-owned"], documents_token(subject="user-1"))
        assert response.status_code == 200
        assert "file-owned" not in file_ids(rows)

    def test_ids_lists_the_callers_files(self, rows, strict_auth):
        response = client.get("/ids", headers=bearer(documents_token(subject="user-1")))
        assert sorted(response.json()) == [
            "file-collided",
            "file-owned",
            "file-untagged",
        ]

    def test_chunks_written_before_tenants_were_recorded_stay_readable(
        self, rows, strict_auth
    ):
        response = get_documents(["file-untagged"], documents_token(subject="user-1"))
        assert response.status_code == 200


class TestScopeIsPartOfTheDeletePredicate:
    """Two owners can hold rows under one file id, so the DELETE must discriminate."""

    def test_only_the_callers_rows_are_removed(self, rows, strict_auth):
        response = delete_documents(
            ["file-collided"], documents_token(subject="user-1")
        )
        assert response.status_code == 200
        survivors = [row for row in rows if row["file_id"] == "file-collided"]
        assert [row["user_id"] for row in survivors] == ["user-2"]

    def test_a_foreign_delete_removes_nothing_at_all(self, rows, strict_auth):
        before = list(rows)
        assert delete_documents(["file-foreign"], documents_token()).status_code == 404
        assert rows == before


class TestTenantSeparation:
    def test_a_file_in_another_tenant_is_not_readable(self, rows, strict_auth):
        token = documents_token(subject="user-1", tenant="tenant-a")
        assert get_documents(["file-other-tenant"], token).status_code == 404

    def test_a_file_in_another_tenant_is_not_deletable(self, rows, strict_auth):
        token = documents_token(subject="user-1", tenant="tenant-a")
        assert delete_documents(["file-other-tenant"], token).status_code == 404
        assert "file-other-tenant" in file_ids(rows)

    def test_the_owning_tenant_still_reads_it(self, rows, strict_auth):
        token = documents_token(subject="user-1", tenant="tenant-b")
        assert get_documents(["file-other-tenant"], token).status_code == 200


class TestEntityScopedFiles:
    """Agent knowledge-base files are reachable exactly as they are on /query."""

    def test_a_permitted_entity_widens_the_scope(self, rows, strict_auth):
        token = documents_token(subject="user-1", entities=("agent-7",))
        response = get_documents(["file-agent"], token, entity_id="agent-7")
        assert response.status_code == 200

    def test_a_permitted_entity_can_be_deleted(self, rows, strict_auth):
        token = documents_token(subject="user-1", entities=("agent-7",))
        response = delete_documents(["file-agent"], token, entity_id="agent-7")
        assert response.status_code == 200
        assert "file-agent" not in file_ids(rows)

    def test_an_unlisted_entity_is_refused(self, rows, strict_auth):
        token = documents_token(subject="user-1")
        response = get_documents(["file-agent"], token, entity_id="agent-7")
        assert response.status_code == 403
        assert "file-agent content" not in response.text

    def test_an_unlisted_entity_cannot_delete(self, rows, strict_auth):
        token = documents_token(subject="user-1")
        assert (
            delete_documents(["file-agent"], token, entity_id="agent-7").status_code
            == 403
        )
        assert "file-agent" in file_ids(rows)

    def test_the_context_route_takes_an_entity_too(self, rows, strict_auth):
        token = documents_token(subject="user-1", entities=("agent-7",))
        response = get_context("file-agent", token, entity_id="agent-7")
        assert response.status_code == 200


class TestTheScopeReachesTheStore:
    """Authorization is the store predicate, not a filter applied to its answer."""

    def test_every_route_passes_its_scope_down(self, rows, strict_auth, monkeypatch):
        seen: List[Dict[str, object]] = []

        async def record_filtered(self, ids, owners=None, tenants=None, executor=None):
            seen.append({"call": "filtered", "owners": owners, "tenants": tenants})
            return list(ids)

        async def record_documents(self, ids, owners=None, tenants=None, executor=None):
            seen.append({"call": "documents", "owners": owners, "tenants": tenants})
            return [Document(page_content="x", metadata={"file_id": ids[0]})]

        async def record_delete(self, ids, owners=None, tenants=None, executor=None):
            seen.append({"call": "delete", "owners": owners, "tenants": tenants})

        monkeypatch.setattr(AsyncPgVector, "get_filtered_ids", record_filtered)
        monkeypatch.setattr(AsyncPgVector, "get_documents_by_ids", record_documents)
        monkeypatch.setattr(AsyncPgVector, "delete_scoped", record_delete)

        token = documents_token(subject="user-1")
        get_documents(["file-owned"], token)
        get_context("file-owned", token)
        delete_documents(["file-owned"], token)

        assert {call["call"] for call in seen} == {"filtered", "documents", "delete"}
        for call in seen:
            assert call["owners"] == ["user-1"]
            assert call["tenants"] == ["__BASE__", None]


EMBED_BODY = {
    "space": space_module.CHAT_SPACE_SPEC.name,
    "input_type": "query",
    "inputs": [{"id": "a", "text": "hello world"}],
}

RERANK_BODY = {
    "profile": "fast-v1",
    "query": "q",
    "candidates": [{"id": "c1", "text": "t", "base_score": 1.0}],
}


@pytest.fixture
def both_planes(monkeypatch, strict_auth):
    """Document routes and the ``/v1`` router, mounted at the same time."""
    monkeypatch.setenv("RAG_SEARCH_API_ENABLED", "true")
    monkeypatch.setenv("RAG_RATE_LIMIT_ENABLED", "false")
    auth.reset_settings()
    install_fake_space(monkeypatch, FakeEmbeddingClient())


def post_embeddings(token):
    return client.post("/v1/embeddings", json=EMBED_BODY, headers=bearer(token))


def post_rerank(token):
    return client.post("/v1/rerank", json=RERANK_BODY, headers=bearer(token))


def reach_every_document_route(token):
    return [
        client.get("/ids", headers=bearer(token)),
        get_documents(["file-owned"], token),
        get_context("file-owned", token),
        delete_documents(["file-owned"], token),
    ]


class TestThePlanesAreSeparable:
    """Reading documents and buying inference are distinct capabilities.

    A token minted to delete a file should not also be spendable against an
    embedding provider, and a token minted to embed should not be able to read
    or destroy stored chunks. Neither scope substitutes for the other, in either
    direction — that is the entire point of splitting them.
    """

    def test_a_documents_token_reaches_every_document_route(self, rows, both_planes):
        token = documents_token(subject="user-1")
        assert [
            response.status_code for response in reach_every_document_route(token)
        ] == [200, 200, 200, 200]

    def test_a_documents_token_buys_no_embedding(self, rows, both_planes):
        response = post_embeddings(documents_token(subject="user-1"))
        assert response.status_code == 403
        assert "rag:embed" in response.json()["detail"]

    def test_a_documents_token_buys_no_rerank(self, rows, both_planes):
        response = post_rerank(documents_token(subject="user-1"))
        assert response.status_code == 403
        assert "rag:rerank" in response.json()["detail"]

    def test_an_embed_token_reaches_no_document_route(self, rows, both_planes):
        token = strict_token(subject="user-1", scopes=("rag:embed",))
        assert post_embeddings(token).status_code == 200
        for response in reach_every_document_route(token):
            assert response.status_code == 403
            assert "rag:documents" in response.json()["detail"]

    def test_a_rerank_token_reaches_no_document_route(self, rows, both_planes):
        token = strict_token(subject="user-1", scopes=("rag:rerank",))
        for response in reach_every_document_route(token):
            assert response.status_code == 403

    def test_an_embed_token_cannot_delete_a_file_it_could_before(
        self, rows, both_planes
    ):
        token = strict_token(subject="user-1", scopes=("rag:embed",))
        assert delete_documents(["file-owned"], token).status_code == 403
        assert "file-owned" in file_ids(rows)

    def test_a_legacy_token_still_reaches_both_planes(self, rows, both_planes):
        """The ``{"id": ...}`` shape predates every scope, including this one."""
        token = legacy_token(user_id="user-1", secret=RAG_SECRET)
        assert [
            response.status_code for response in reach_every_document_route(token)
        ] == [200, 200, 200, 200]
        assert post_embeddings(token).status_code == 200

    def test_an_application_signed_legacy_token_still_reaches_the_document_routes(
        self, rows, both_planes
    ):
        """``RAG_AUTH_ACCEPT_LEGACY=true`` keeps today's callers working."""
        token = legacy_token(user_id="user-1", secret=APP_SECRET)
        assert [
            response.status_code for response in reach_every_document_route(token)
        ] == [200, 200, 200, 200]
