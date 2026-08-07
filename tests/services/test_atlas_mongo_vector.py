"""AtlasMongoVector: the scoped file lookups and the indexes they depend on.

The Atlas vector-search index accelerates ``$vectorSearch`` only. Every lookup
here is an ordinary ``find``, so the standard indexes are part of the deployment
contract rather than a tuning suggestion — and the contract lives in README.md,
where it drifts silently. These tests pin both halves to the same source.
"""

import re
from pathlib import Path
from typing import Any, Dict, List

import pytest

from app.services.vector_store.atlas_mongo_vector import AtlasMongoVector

BASE_TENANTS = ["__BASE__", None]

ROWS = [
    {
        "_id": "file-owned_d1",
        "file_id": "file-owned",
        "user_id": "user-1",
        "tenant_id": "__BASE__",
        "digest": "d1",
        "text": "owned chunk",
        "source": "a.txt",
    },
    {
        "_id": "file-foreign_d2",
        "file_id": "file-foreign",
        "user_id": "user-2",
        "tenant_id": "__BASE__",
        "digest": "d2",
        "text": "foreign chunk",
        "source": "b.txt",
    },
    {
        "_id": "file-collided_d3",
        "file_id": "file-collided",
        "user_id": "user-2",
        "tenant_id": "__BASE__",
        "digest": "d3",
        "text": "theirs",
        "source": "c.txt",
    },
    {
        "_id": "file-collided_d4",
        "file_id": "file-collided",
        "user_id": "user-1",
        "tenant_id": "__BASE__",
        "digest": "d4",
        "text": "mine",
        "source": "d.txt",
    },
]


def _matches(row: Dict[str, Any], predicate: Dict[str, Any]) -> bool:
    """Minimal evaluator for the MongoDB query dialect these methods emit."""
    for key, clause in predicate.items():
        if key == "$and":
            if not all(_matches(row, sub) for sub in clause):
                return False
            continue
        if key == "$or":
            if not any(_matches(row, sub) for sub in clause):
                return False
            continue
        operator, operand = next(iter(clause.items()))
        if operator != "$in":
            raise AssertionError(f"unexpected operator: {operator}")
        if row.get(key) not in operand:
            return False
    return True


class FakeCollection:
    """In-memory stand-in that evaluates the predicate it is handed."""

    def __init__(self, rows):
        self.rows = [dict(row) for row in rows]
        self.indexes: List[Any] = []

    def create_index(self, keys, name=None, **kwargs):
        self.indexes.append({"keys": list(keys), "name": name})
        return name

    def find(self, predicate, projection=None):
        return [dict(row) for row in self.rows if _matches(row, predicate)]

    def distinct(self, field, predicate=None):
        return sorted(
            {
                row[field]
                for row in self.rows
                if predicate is None or _matches(row, predicate)
            }
        )

    def delete_many(self, predicate):
        self.rows = [row for row in self.rows if not _matches(row, predicate)]


class Store(AtlasMongoVector):
    """Bypasses MongoDBAtlasVectorSearch.__init__, which wants a live client."""

    def __init__(self, collection):
        self._collection = collection

    def indexes(self):
        return self._collection.indexes


@pytest.fixture
def store():
    return Store(FakeCollection(ROWS))


def contents(documents):
    return sorted(document.page_content for document in documents)


class TestFileLookupsAreScoped:
    def test_a_foreign_file_resolves_to_nothing(self, store):
        assert store.get_filtered_ids(["file-foreign"], ["user-1"], BASE_TENANTS) == []
        assert (
            store.get_documents_by_ids(["file-foreign"], ["user-1"], BASE_TENANTS) == []
        )

    def test_the_callers_own_file_resolves(self, store):
        assert store.get_filtered_ids(["file-owned"], ["user-1"], BASE_TENANTS) == [
            "file-owned"
        ]
        documents = store.get_documents_by_ids(["file-owned"], ["user-1"], BASE_TENANTS)
        assert contents(documents) == ["owned chunk"]

    def test_a_collided_file_id_yields_only_the_callers_chunks(self, store):
        documents = store.get_documents_by_ids(
            ["file-collided"], ["user-1"], BASE_TENANTS
        )
        assert contents(documents) == ["mine"]

    def test_another_tenant_sees_nothing(self, store):
        assert (
            store.get_documents_by_ids(["file-owned"], ["user-1"], ["tenant-b"]) == []
        )

    def test_listing_is_scoped_to_the_caller(self, store):
        assert store.get_all_ids(["user-1"], BASE_TENANTS) == [
            "file-collided",
            "file-owned",
        ]

    def test_an_empty_owner_set_authorizes_nothing(self, store):
        assert store.get_all_ids([], BASE_TENANTS) == []
        assert store.get_documents_by_ids(["file-owned"], [], BASE_TENANTS) == []


class TestDeletesAreScoped:
    def test_a_foreign_file_is_not_deleted(self, store):
        store.delete_scoped(["file-foreign"], ["user-1"], BASE_TENANTS)
        survivors = store.get_documents_by_ids(
            ["file-foreign"], ["user-2"], BASE_TENANTS
        )
        assert contents(survivors) == ["foreign chunk"]

    def test_a_collided_file_id_deletes_only_the_callers_rows(self, store):
        store.delete_scoped(["file-collided"], ["user-1"], BASE_TENANTS)
        survivors = store.get_documents_by_ids(
            ["file-collided"], ["user-2"], BASE_TENANTS
        )
        assert contents(survivors) == ["theirs"]

    def test_the_callers_own_file_is_deleted(self, store):
        store.delete_scoped(["file-owned"], ["user-1"], BASE_TENANTS)
        assert (
            store.get_documents_by_ids(["file-owned"], ["user-1"], BASE_TENANTS) == []
        )

    def test_an_empty_owner_set_deletes_nothing(self, store):
        store.delete_scoped(["file-owned"], [], BASE_TENANTS)
        assert store.get_documents_by_ids(["file-owned"], ["user-1"], BASE_TENANTS)


class TestTheIndexesTheLookupsNeed:
    """Every ordinary ``find`` this store makes has to have an index behind it."""

    def documented_indexes(self):
        readme = Path(__file__).resolve().parents[2] / "README.md"
        keys = set()
        for spec in re.findall(
            r"createIndex\(\s*(\{.*?\})\s*\)", readme.read_text(encoding="utf-8")
        ):
            keys.add(tuple(re.findall(r"(\w+)\s*:\s*1", spec)))
        return keys

    def test_startup_creates_the_file_id_index(self, store):
        store.ensure_indexes()
        created = {tuple(key for key, _ in index["keys"]) for index in store.indexes()}
        assert ("file_id",) in created

    def test_startup_creates_the_digest_index_the_rerank_probe_needs(self, store):
        """Candidate ids are normally digests, and each rerank resolves them twice."""
        store.ensure_indexes()
        created = {tuple(key for key, _ in index["keys"]) for index in store.indexes()}
        assert ("digest", "user_id", "tenant_id") in created

    def test_the_created_indexes_match_the_documented_ones(self, store):
        store.ensure_indexes()
        created = {tuple(key for key, _ in index["keys"]) for index in store.indexes()}
        assert created == self.documented_indexes()

    def test_every_index_is_named_so_it_is_recognisable(self, store):
        store.ensure_indexes()
        assert all(index["name"] for index in store.indexes())
