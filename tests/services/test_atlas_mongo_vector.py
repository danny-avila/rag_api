"""The owner predicate holds on the atlas-mongo store too.

There is no Atlas instance in CI, so the collection here is a small in-memory
double that *evaluates* the filter documents rather than recording them: the
thing under test is the predicate each method builds, and a double that only
asserted on call arguments would pass just as happily against a filter that
selects the wrong rows.

The vector search path is covered by asserting the `pre_filter` handed to Atlas,
since scoring is Atlas's own and cannot be reproduced here.
"""

from typing import Any, Dict, List, Optional

import pytest

from app.scope import ScopeFilter, file_clause
from app.services.vector_store.atlas_mongo_vector import AtlasMongoVector

VICTIM = "victim-user"
ATTACKER = "attacker-user"
AGENT = "agent-abc"


def _matches(document: Dict[str, Any], query: Dict[str, Any]) -> bool:
    """Evaluate the subset of the query language these methods use."""
    for field, condition in query.items():
        if field == "$and":
            if not all(_matches(document, clause) for clause in condition):
                return False
            continue
        value = document.get(field)
        if isinstance(condition, dict):
            if "$in" in condition and value not in condition["$in"]:
                return False
            if "$eq" in condition and value != condition["$eq"]:
                return False
        elif value != condition:
            return False
    return True


class FakeCollection:
    """Enough of a pymongo collection for the store's four scoped methods."""

    def __init__(self, documents: List[Dict[str, Any]]):
        self.documents = list(documents)

    def distinct(self, field: str, query: Optional[Dict[str, Any]] = None) -> List[str]:
        query = query or {}
        seen = []
        for document in self.documents:
            if not _matches(document, query):
                continue
            value = document.get(field)
            if value is not None and value not in seen:
                seen.append(value)
        return seen

    def find(self, query: Optional[Dict[str, Any]] = None):
        query = query or {}
        return [d for d in self.documents if _matches(d, query)]

    def delete_many(self, query: Dict[str, Any]) -> None:
        self.documents = [d for d in self.documents if not _matches(d, query)]


def _chunk(file_id: str, owner: Optional[str], text: str) -> Dict[str, Any]:
    document = {
        "file_id": file_id,
        "text": text,
        "digest": f"digest-{text}",
        "source": "test",
        "page": 0,
    }
    if owner is not None:
        document["user_id"] = owner
    return document


@pytest.fixture()
def store():
    instance = AtlasMongoVector.__new__(AtlasMongoVector)
    instance._collection = FakeCollection(
        [
            _chunk("file-victim", VICTIM, "victim secret"),
            _chunk("file-attacker", ATTACKER, "attacker note"),
            _chunk("file-agent", AGENT, "agent knowledge"),
            # One caller-chosen file id held by two owners.
            _chunk("file-shared", ATTACKER, "attacker own row"),
            _chunk("file-shared", VICTIM, "victim hidden row"),
            # Written before owners were recorded.
            _chunk("file-orphan", None, "unowned row"),
        ]
    )
    return instance


def test_get_all_ids_returns_only_the_callers_files(store):
    assert sorted(store.get_all_ids(owners=[ATTACKER])) == [
        "file-attacker",
        "file-shared",
    ]


def test_get_filtered_ids_is_not_an_existence_oracle(store):
    assert store.get_filtered_ids(["file-victim"], owners=[ATTACKER]) == []
    assert store.get_filtered_ids(["file-victim"], owners=[VICTIM]) == ["file-victim"]


def test_get_documents_by_ids_refuses_foreign_content(store):
    assert store.get_documents_by_ids(["file-victim"], owners=[ATTACKER]) == []

    owned = store.get_documents_by_ids(["file-victim"], owners=[VICTIM])
    assert [document.page_content for document in owned] == ["victim secret"]


def test_one_file_id_held_by_two_owners_stays_separated(store):
    attacker_docs = store.get_documents_by_ids(["file-shared"], owners=[ATTACKER])
    victim_docs = store.get_documents_by_ids(["file-shared"], owners=[VICTIM])

    assert [d.page_content for d in attacker_docs] == ["attacker own row"]
    assert [d.page_content for d in victim_docs] == ["victim hidden row"]


def test_delete_scoped_cannot_remove_foreign_chunks(store):
    store.delete_scoped(["file-victim"], owners=[ATTACKER])
    assert store.get_documents_by_ids(["file-victim"], owners=[VICTIM]) != []

    store.delete_scoped(["file-shared"], owners=[ATTACKER])
    assert store.get_documents_by_ids(["file-shared"], owners=[ATTACKER]) == []
    # The other owner's rows under the same file id survive.
    assert store.get_documents_by_ids(["file-shared"], owners=[VICTIM]) != []


def test_chunks_with_no_owner_are_not_readable_by_anyone(store):
    assert store.get_documents_by_ids(["file-orphan"], owners=[ATTACKER]) == []
    assert store.get_documents_by_ids(["file-orphan"], owners=[VICTIM]) == []
    assert "file-orphan" not in store.get_all_ids(owners=[ATTACKER])


def test_empty_owner_set_reads_nothing(store):
    """A scope with no owners must not degrade into "no filter"."""
    assert store.get_all_ids(owners=[]) == []
    assert store.get_filtered_ids(["file-victim"], owners=[]) == []
    assert store.get_documents_by_ids(["file-victim"], owners=[]) == []

    store.delete_scoped(["file-victim"], owners=[])
    assert store.get_documents_by_ids(["file-victim"], owners=[VICTIM]) != []


def test_vector_search_sends_the_owner_clause_as_a_pre_filter(store, monkeypatch):
    """Atlas applies `pre_filter` inside `$vectorSearch`, so the owner clause has
    to reach it — scoring the whole file and trimming afterwards would both leak
    and consume `k`."""
    captured = {}

    def fake_search(embedding, k, pre_filter, post_filter_pipeline, **kwargs):
        captured["pre_filter"] = pre_filter
        return []

    monkeypatch.setattr(store, "_similarity_search_with_score", fake_search)

    scope = ScopeFilter(owners=(ATTACKER,))
    store.similarity_search_with_score_by_vector(
        [0.1, 0.2, 0.3], k=4, filter=scope.predicate(file_clause("file-shared"))
    )

    assert captured["pre_filter"] == {
        "$and": [
            {"file_id": {"$eq": "file-shared"}},
            {"user_id": {"$in": [ATTACKER]}},
        ]
    }
