"""Integration tests: the owner predicate holds against a real pgvector database.

Every test here corresponds to a hole that was open before this change. They run
against a real PostgreSQL+pgvector container and a real ``ExtendedPgVector``, so
what is proven is the SQL that ships — not a double's idea of it.

Run with:  pytest tests/integration/ -m integration -v
"""

import json
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.scope import ScopeFilter, file_clause, files_clause
from app.services.vector_store.extended_pg_vector import ExtendedPgVector

pytestmark = pytest.mark.integration

VICTIM = "victim-user"
ATTACKER = "attacker-user"
AGENT = "agent-abc"


def _seed(conn, collection_id, *, file_id, owner, count=3, prefix="chunk"):
    """Insert ``count`` chunks of ``file_id`` owned by ``owner``.

    ``owner=None`` writes a row with no ``user_id`` at all — the shape that read
    as "belongs to everyone" before this change.
    """
    metadata = {"file_id": file_id}
    if owner is not None:
        metadata["user_id"] = owner
    for i in range(count):
        conn.execute(
            text(
                "INSERT INTO langchain_pg_embedding "
                "(collection_id, embedding, document, cmetadata, custom_id) "
                "VALUES (:cid, :emb, :doc, :meta, :cust)"
            ),
            {
                "cid": collection_id,
                "emb": f"[{0.1 * (i + 1)},{0.2 * (i + 1)},{0.3 * (i + 1)}]",
                "doc": f"{prefix}-{owner}-{i}",
                "meta": json.dumps({**metadata, "index": i}),
                "cust": file_id,
            },
        )


@pytest.fixture()
def store(engine):
    from langchain_community.vectorstores.pgvector import (
        _get_embedding_collection_store,
    )

    class TestableStore(ExtendedPgVector):
        def __init__(self):
            self._bind = engine
            EmbeddingStore, CollectionStore = _get_embedding_collection_store(
                vector_dimension=3, use_jsonb=True
            )
            self.EmbeddingStore = EmbeddingStore
            self.CollectionStore = CollectionStore
            self.collection_name = "test_collection"
            self.collection_metadata = None

    return TestableStore()


@pytest.fixture()
def two_owners(engine, collection_id):
    """Victim and attacker each hold a distinct file; ids are returned."""
    victim_file = f"file-victim-{uuid.uuid4().hex[:8]}"
    attacker_file = f"file-attacker-{uuid.uuid4().hex[:8]}"
    with engine.begin() as conn:
        _seed(conn, collection_id, file_id=victim_file, owner=VICTIM, prefix="secret")
        _seed(conn, collection_id, file_id=attacker_file, owner=ATTACKER)
    yield victim_file, attacker_file
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM langchain_pg_embedding WHERE custom_id = ANY(:ids)"),
            {"ids": [victim_file, attacker_file]},
        )


def test_get_all_ids_returns_only_the_callers_files(store, two_owners):
    """``GET /ids`` listed every file in the deployment, which is how an
    attacker discovered the file ids to then read."""
    victim_file, attacker_file = two_owners

    ids = store.get_all_ids(owners=[ATTACKER])

    assert attacker_file in ids
    assert victim_file not in ids


def test_get_filtered_ids_is_not_an_existence_oracle(store, two_owners):
    """Naming a foreign file id answered "it exists"."""
    victim_file, _ = two_owners

    assert store.get_filtered_ids([victim_file], owners=[ATTACKER]) == []
    assert store.get_filtered_ids([victim_file], owners=[VICTIM]) == [victim_file]


def test_get_documents_by_ids_refuses_foreign_content(store, two_owners):
    """``GET /documents`` and ``/documents/{id}/context`` returned the chunks of
    any file id the caller could name."""
    victim_file, _ = two_owners

    assert store.get_documents_by_ids([victim_file], owners=[ATTACKER]) == []

    owned = store.get_documents_by_ids([victim_file], owners=[VICTIM])
    assert len(owned) == 3
    assert all("secret" in doc.page_content for doc in owned)


def test_delete_scoped_cannot_remove_foreign_chunks(store, two_owners, engine):
    """``DELETE /documents`` deleted by file id alone, so naming a victim's file
    id destroyed their chunks."""
    victim_file, _ = two_owners

    store.delete_scoped([victim_file], owners=[ATTACKER])

    with engine.begin() as conn:
        surviving = conn.execute(
            text("SELECT count(*) FROM langchain_pg_embedding WHERE custom_id = :fid"),
            {"fid": victim_file},
        ).scalar()
    assert surviving == 3

    store.delete_scoped([victim_file], owners=[VICTIM])
    with engine.begin() as conn:
        remaining = conn.execute(
            text("SELECT count(*) FROM langchain_pg_embedding WHERE custom_id = :fid"),
            {"fid": victim_file},
        ).scalar()
    assert remaining == 0


def test_one_file_id_held_by_two_owners_stays_separated(store, engine, collection_id):
    """A file id is chosen by whoever uploads, so two owners can hold rows under
    the same id. Each may read only their own."""
    shared_id = f"file-collision-{uuid.uuid4().hex[:8]}"
    with engine.begin() as conn:
        _seed(conn, collection_id, file_id=shared_id, owner=VICTIM, prefix="victim")
        _seed(conn, collection_id, file_id=shared_id, owner=ATTACKER, prefix="attacker")

    try:
        victim_docs = store.get_documents_by_ids([shared_id], owners=[VICTIM])
        attacker_docs = store.get_documents_by_ids([shared_id], owners=[ATTACKER])

        assert len(victim_docs) == 3
        assert len(attacker_docs) == 3
        assert all("victim" in doc.page_content for doc in victim_docs)
        assert all("attacker" in doc.page_content for doc in attacker_docs)

        store.delete_scoped([shared_id], owners=[ATTACKER])
        assert len(store.get_documents_by_ids([shared_id], owners=[VICTIM])) == 3
    finally:
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM langchain_pg_embedding WHERE custom_id = :fid"),
                {"fid": shared_id},
            )


def test_chunks_with_no_owner_are_not_readable_by_anyone(store, engine, collection_id):
    """An absent ``user_id`` read as "belongs to everyone", so any such chunk was
    readable by every caller. It is now owned by nobody instead."""
    orphan_file = f"file-orphan-{uuid.uuid4().hex[:8]}"
    with engine.begin() as conn:
        _seed(conn, collection_id, file_id=orphan_file, owner=None)

    try:
        assert store.get_documents_by_ids([orphan_file], owners=[ATTACKER]) == []
        assert store.get_documents_by_ids([orphan_file], owners=[VICTIM]) == []
        assert store.get_all_ids(owners=[ATTACKER]) == [] or (
            orphan_file not in store.get_all_ids(owners=[ATTACKER])
        )
    finally:
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM langchain_pg_embedding WHERE custom_id = :fid"),
                {"fid": orphan_file},
            )


def test_empty_owner_set_reads_nothing(store, two_owners):
    """A scope with no owners must not degrade into "no filter"."""
    victim_file, attacker_file = two_owners

    assert store.get_all_ids(owners=[]) == []
    assert store.get_filtered_ids([victim_file, attacker_file], owners=[]) == []
    assert store.get_documents_by_ids([victim_file, attacker_file], owners=[]) == []


def test_scope_predicate_filters_ranked_search(store, engine, collection_id):
    """``/query`` authorized a whole result set from ``documents[0]``, so any hit
    past the first was never checked. The predicate now runs before ranking.
    """
    shared_id = f"file-ranked-{uuid.uuid4().hex[:8]}"
    with engine.begin() as conn:
        # Attacker's row sorts nearest the probe vector, so under the old
        # first-hit check it would authorize the victim's rows behind it.
        _seed(
            conn,
            collection_id,
            file_id=shared_id,
            owner=ATTACKER,
            count=1,
            prefix="attacker",
        )
        _seed(conn, collection_id, file_id=shared_id, owner=VICTIM, prefix="victim")

    try:
        scope = ScopeFilter(owners=(ATTACKER,))
        clause = store._create_filter_clause(scope.predicate(file_clause(shared_id)))

        with Session(engine) as session:
            rows = (
                session.query(store.EmbeddingStore.document)
                .filter(store.EmbeddingStore.collection_id == collection_id)
                .filter(clause)
                .all()
            )

        documents = [row[0] for row in rows]
        assert documents
        assert all("attacker" in doc for doc in documents)
        assert not any("victim" in doc for doc in documents)
    finally:
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM langchain_pg_embedding WHERE custom_id = :fid"),
                {"fid": shared_id},
            )


def test_entity_id_widens_rather_than_replaces_the_owner(store, engine, collection_id):
    """An agent knowledge base is owned by the agent id, so a caller reading it
    must reach both owners — and only those two."""
    own_file = f"file-own-{uuid.uuid4().hex[:8]}"
    agent_file = f"file-agent-{uuid.uuid4().hex[:8]}"
    victim_file = f"file-victim-{uuid.uuid4().hex[:8]}"
    with engine.begin() as conn:
        _seed(conn, collection_id, file_id=own_file, owner=ATTACKER)
        _seed(conn, collection_id, file_id=agent_file, owner=AGENT)
        _seed(conn, collection_id, file_id=victim_file, owner=VICTIM)

    try:
        scope = ScopeFilter(owners=tuple(sorted({ATTACKER, AGENT})))
        ids = store.get_filtered_ids(
            [own_file, agent_file, victim_file], owners=scope.owners
        )

        assert set(ids) == {own_file, agent_file}
    finally:
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM langchain_pg_embedding WHERE custom_id = ANY(:ids)"),
                {"ids": [own_file, agent_file, victim_file]},
            )


def test_multi_file_query_predicate_drops_foreign_files(store, engine, collection_id):
    """``/query_multiple`` performed no authorization at all: every file id the
    caller listed was searched and returned."""
    mine = f"file-mine-{uuid.uuid4().hex[:8]}"
    theirs = f"file-theirs-{uuid.uuid4().hex[:8]}"
    with engine.begin() as conn:
        _seed(conn, collection_id, file_id=mine, owner=ATTACKER, prefix="mine")
        _seed(conn, collection_id, file_id=theirs, owner=VICTIM, prefix="theirs")

    try:
        scope = ScopeFilter(owners=(ATTACKER,))
        clause = store._create_filter_clause(
            scope.predicate(files_clause([mine, theirs]))
        )

        with Session(engine) as session:
            rows = (
                session.query(store.EmbeddingStore.document)
                .filter(store.EmbeddingStore.collection_id == collection_id)
                .filter(clause)
                .all()
            )

        documents = [row[0] for row in rows]
        assert documents
        assert all("mine" in doc for doc in documents)
        assert not any("theirs" in doc for doc in documents)
    finally:
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM langchain_pg_embedding WHERE custom_id = ANY(:ids)"),
                {"ids": [mine, theirs]},
            )


def test_sibling_collection_rows_are_not_visible(store, engine, collection_id):
    """``langchain_pg_embedding`` is shared by every collection, so a lookup that
    omitted ``collection_id`` could read rows this store does not serve."""
    other_file = f"file-other-collection-{uuid.uuid4().hex[:8]}"
    with engine.begin() as conn:
        other_collection = conn.execute(
            text(
                "INSERT INTO langchain_pg_collection (name, cmetadata) "
                "VALUES (:name, :meta) RETURNING uuid"
            ),
            {"name": f"other-{uuid.uuid4().hex[:8]}", "meta": "{}"},
        ).fetchone()[0]
        _seed(conn, collection_id=other_collection, file_id=other_file, owner=ATTACKER)

    try:
        assert store.get_filtered_ids([other_file], owners=[ATTACKER]) == []
        assert store.get_documents_by_ids([other_file], owners=[ATTACKER]) == []
        assert other_file not in store.get_all_ids(owners=[ATTACKER])
    finally:
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM langchain_pg_embedding WHERE custom_id = :fid"),
                {"fid": other_file},
            )
            conn.execute(
                text("DELETE FROM langchain_pg_collection WHERE uuid = :cid"),
                {"cid": other_collection},
            )
