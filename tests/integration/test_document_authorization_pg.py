"""File-addressed reads and deletes against real PostgreSQL.

``GET /documents``, ``GET /documents/{id}/context`` and ``DELETE /documents``
address rows by a caller-supplied file id. These prove the owner and tenant
predicate is in the SQL — not applied to the rows afterwards — including the
case a fake store cannot show: two owners holding rows under the same file id,
where an unscoped ``DELETE ... WHERE custom_id IN (...)`` destroys both.
"""

import uuid

import pytest
from langchain_core.documents import Document

from app.services.vector_store.async_pg_vector import AsyncPgVector
from tests.integration.conftest import DeterministicEmbeddings

pytestmark = pytest.mark.integration

BASE_TENANTS = ["__BASE__", None]

# (file_id, user_id, tenant_id, chunk)
CORPUS = [
    ("file-owned", "user-1", "__BASE__", "owned chunk"),
    ("file-foreign", "user-2", "__BASE__", "foreign chunk"),
    ("file-agent", "agent-7", "__BASE__", "agent chunk"),
    ("file-other-tenant", "user-1", "tenant-b", "other tenant chunk"),
    ("file-collided", "user-1", "__BASE__", "mine"),
    ("file-collided", "user-2", "__BASE__", "theirs"),
    ("file-untagged", "user-1", None, "pre-tenant chunk"),
]


@pytest.fixture()
def seeded(pg_store):
    for file_id, user_id, tenant_id, chunk in CORPUS:
        metadata = {"file_id": file_id, "user_id": user_id}
        if tenant_id is not None:
            metadata["tenant_id"] = tenant_id
        pg_store.add_documents(
            [Document(page_content=chunk, metadata=metadata)], ids=[file_id]
        )
    return pg_store


@pytest.fixture()
def sibling_collection(pg_url, engine, _create_tables, monkeypatch):
    """A second collection sharing ``langchain_pg_embedding``."""
    from app.services.vector_store.factory import get_vector_store
    from tests.conftest import ORIGINAL_PGVECTOR_POST_INIT

    monkeypatch.setattr(AsyncPgVector, "__post_init__", ORIGINAL_PGVECTOR_POST_INIT)
    store = get_vector_store(
        connection_string=pg_url,
        embeddings=DeterministicEmbeddings(),
        collection_name=f"sibling-{uuid.uuid4().hex}",
        mode="async",
        create_extension=False,
    )
    yield store
    store._bind.dispose()


def contents(documents):
    return sorted(document.page_content for document in documents)


class TestReadsAreScopedInSql:
    async def test_a_foreign_file_resolves_to_nothing(self, seeded):
        assert (
            await seeded.get_filtered_ids(["file-foreign"], ["user-1"], BASE_TENANTS)
            == []
        )
        assert (
            await seeded.get_documents_by_ids(
                ["file-foreign"], ["user-1"], BASE_TENANTS
            )
            == []
        )

    async def test_the_owners_own_file_still_resolves(self, seeded):
        assert await seeded.get_filtered_ids(
            ["file-owned"], ["user-1"], BASE_TENANTS
        ) == ["file-owned"]
        documents = await seeded.get_documents_by_ids(
            ["file-owned"], ["user-1"], BASE_TENANTS
        )
        assert contents(documents) == ["owned chunk"]

    async def test_a_collided_file_id_yields_only_the_callers_chunks(self, seeded):
        documents = await seeded.get_documents_by_ids(
            ["file-collided"], ["user-1"], BASE_TENANTS
        )
        assert contents(documents) == ["mine"]

    async def test_another_tenant_does_not_see_the_same_owners_rows(self, seeded):
        assert (
            await seeded.get_documents_by_ids(
                ["file-other-tenant"], ["user-1"], ["tenant-a"]
            )
            == []
        )
        documents = await seeded.get_documents_by_ids(
            ["file-other-tenant"], ["user-1"], ["tenant-b"]
        )
        assert contents(documents) == ["other tenant chunk"]

    async def test_the_base_tenant_matches_chunks_written_before_tenants(self, seeded):
        documents = await seeded.get_documents_by_ids(
            ["file-untagged"], ["user-1"], BASE_TENANTS
        )
        assert contents(documents) == ["pre-tenant chunk"]

    async def test_a_permitted_entity_widens_the_owner_set(self, seeded):
        documents = await seeded.get_documents_by_ids(
            ["file-agent"], ["agent-7", "user-1"], BASE_TENANTS
        )
        assert contents(documents) == ["agent chunk"]

    async def test_an_empty_owner_set_authorizes_nothing(self, seeded):
        assert await seeded.get_documents_by_ids(["file-owned"], [], BASE_TENANTS) == []

    async def test_listing_is_scoped_to_the_caller(self, seeded):
        assert sorted(await seeded.get_all_ids(["user-1"], BASE_TENANTS)) == [
            "file-collided",
            "file-owned",
            "file-untagged",
        ]

    async def test_a_sibling_collections_rows_are_invisible(
        self, seeded, sibling_collection
    ):
        await sibling_collection.aadd_documents(
            [
                Document(
                    page_content="sibling chunk",
                    metadata={
                        "file_id": "file-owned",
                        "user_id": "user-1",
                        "tenant_id": "__BASE__",
                    },
                )
            ],
            ids=["file-owned"],
        )
        documents = await seeded.get_documents_by_ids(
            ["file-owned"], ["user-1"], BASE_TENANTS
        )
        assert contents(documents) == ["owned chunk"]


class TestDeletesAreScopedInSql:
    async def test_a_foreign_file_is_not_deleted(self, seeded):
        await seeded.delete_scoped(["file-foreign"], ["user-1"], BASE_TENANTS)
        survivors = await seeded.get_documents_by_ids(
            ["file-foreign"], ["user-2"], BASE_TENANTS
        )
        assert contents(survivors) == ["foreign chunk"]

    async def test_the_owners_own_file_is_deleted(self, seeded):
        await seeded.delete_scoped(["file-owned"], ["user-1"], BASE_TENANTS)
        assert (
            await seeded.get_documents_by_ids(["file-owned"], ["user-1"], BASE_TENANTS)
            == []
        )

    async def test_a_collided_file_id_deletes_only_the_callers_rows(self, seeded):
        """The case the separate existence check cannot cover.

        Both owners hold rows under ``file-collided``; a DELETE that filters on
        the file id alone removes the other owner's chunk as well.
        """
        await seeded.delete_scoped(["file-collided"], ["user-1"], BASE_TENANTS)
        survivors = await seeded.get_documents_by_ids(
            ["file-collided"], ["user-2"], BASE_TENANTS
        )
        assert contents(survivors) == ["theirs"]

    async def test_another_tenant_cannot_delete_the_same_owners_rows(self, seeded):
        await seeded.delete_scoped(["file-other-tenant"], ["user-1"], ["tenant-a"])
        survivors = await seeded.get_documents_by_ids(
            ["file-other-tenant"], ["user-1"], ["tenant-b"]
        )
        assert contents(survivors) == ["other tenant chunk"]

    async def test_an_empty_owner_set_deletes_nothing(self, seeded):
        await seeded.delete_scoped(["file-owned"], [], BASE_TENANTS)
        documents = await seeded.get_documents_by_ids(
            ["file-owned"], ["user-1"], BASE_TENANTS
        )
        assert contents(documents) == ["owned chunk"]

    async def test_a_sibling_collections_rows_survive(self, seeded, sibling_collection):
        await sibling_collection.aadd_documents(
            [
                Document(
                    page_content="sibling chunk",
                    metadata={
                        "file_id": "file-owned",
                        "user_id": "user-1",
                        "tenant_id": "__BASE__",
                    },
                )
            ],
            ids=["file-owned"],
        )
        await seeded.delete_scoped(["file-owned"], ["user-1"], BASE_TENANTS)
        survivors = await sibling_collection.get_documents_by_ids(
            ["file-owned"], ["user-1"], BASE_TENANTS
        )
        assert contents(survivors) == ["sibling chunk"]
