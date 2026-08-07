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

# (file_id, user_id, chunk, tenant_id) — a None tenant models a chunk written
# before tenants were recorded on write.
CORPUS = [
    ("file-owned", "user-1", "owner chunk one", None),
    ("file-owned", "user-1", "owner chunk two", None),
    ("file-foreign", "user-2", "foreign chunk", None),
    ("file-shared", "user-1", "shared chunk owned by user one", None),
    ("file-shared", "user-2", "shared chunk owned by user two", None),
    ("file-agent", "agent-7", "agent knowledge chunk", None),
    ("file-tenanted", "user-1", "chunk in tenant a", "tenant-a"),
    ("file-tenanted", "user-1", "chunk in tenant b", "tenant-b"),
]


def digest(text_value: str) -> str:
    return hashlib.md5(text_value.encode("utf-8")).hexdigest()


@pytest.fixture()
def seeded(pg_store, monkeypatch):
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


def query(file_id, subject="user-1", entity_id=None, token=None, k=20, tenant=None):
    body = {"query": "chunk", "file_id": file_id, "k": k}
    if entity_id:
        body["entity_id"] = entity_id
    if token is None:
        token = (
            strict_token(subject=subject, tenant=tenant)
            if tenant
            else strict_token(subject=subject)
        )
    return client.post("/query", json=body, headers=bearer(token))


def contents_of(response):
    return {hit[0]["page_content"] for hit in response.json()}


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


class TestTenantScopeInPostgres:
    """Tenant is enforced by the SQL predicate, not by a later filter."""

    def test_a_tenant_sees_only_its_own_side_of_a_shared_file(self, seeded):
        response = query("file-tenanted", tenant="tenant-a")
        assert response.status_code == 200
        assert contents_of(response) == {"chunk in tenant a"}

    def test_the_other_tenant_sees_only_its_own_side(self, seeded):
        response = query("file-tenanted", tenant="tenant-b")
        assert response.status_code == 200
        assert contents_of(response) == {"chunk in tenant b"}

    def test_the_base_tenant_sees_neither(self, seeded):
        response = query("file-tenanted", tenant="__BASE__")
        assert response.status_code == 200
        assert response.json() == []

    def test_the_base_tenant_still_reads_chunks_written_before_tenants_existed(
        self, seeded
    ):
        response = query("file-owned", tenant="__BASE__")
        assert response.status_code == 200
        assert len(response.json()) == 2

    def test_a_real_tenant_never_absorbs_untagged_chunks(self, seeded):
        response = query("file-owned", tenant="tenant-a")
        assert response.status_code == 200
        assert response.json() == []

    def test_the_system_tenant_is_refused(self, seeded):
        response = query("file-owned", tenant="__SYSTEM__")
        assert response.status_code == 403

    def test_query_multiple_is_tenant_scoped_in_sql(self, seeded):
        response = client.post(
            "/query_multiple",
            json={
                "query": "chunk",
                "file_ids": ["file-owned", "file-tenanted"],
                "k": 20,
            },
            headers=bearer(strict_token(subject="user-1", tenant="tenant-a")),
        )
        assert response.status_code == 200
        assert contents_of(response) == {"chunk in tenant a"}


class TestWritePathRecordsScope:
    """Chunks are stamped with the writer's tenant so reads can scope on it."""

    def test_uploaded_chunks_record_the_callers_tenant(self, seeded, tmp_path, engine):
        from sqlalchemy import text

        upload = tmp_path / "note.txt"
        upload.write_text("a freshly uploaded chunk of text")
        with upload.open("rb") as handle:
            response = client.post(
                "/embed",
                data={"file_id": "file-uploaded"},
                files={"file": ("note.txt", handle, "text/plain")},
                headers=bearer(strict_token(subject="user-9", tenant="tenant-z")),
            )
        assert response.status_code == 200, response.text

        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT cmetadata->>'user_id', cmetadata->>'tenant_id' "
                    "FROM langchain_pg_embedding WHERE custom_id = :file_id"
                ),
                {"file_id": "file-uploaded"},
            ).fetchall()
        assert rows
        assert all(row[0] == "user-9" and row[1] == "tenant-z" for row in rows)

    def test_writing_into_an_unpermitted_entity_is_refused(self, seeded, tmp_path):
        upload = tmp_path / "poison.txt"
        upload.write_text("content aimed at another entity")
        with upload.open("rb") as handle:
            response = client.post(
                "/embed",
                data={"file_id": "file-poison", "entity_id": "agent-7"},
                files={"file": ("poison.txt", handle, "text/plain")},
                headers=bearer(strict_token(subject="user-9")),
            )
        assert response.status_code == 403
