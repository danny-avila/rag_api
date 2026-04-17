import pytest
import sqlalchemy
from sqlalchemy.dialects import postgresql
from langchain_community.vectorstores.pgvector import _get_embedding_collection_store

from app.services.vector_store.extended_pg_vector import ExtendedPgVector


class DummyPgVector(ExtendedPgVector):
    def __init__(self):
        self._bind = None
        EmbeddingStore, _ = _get_embedding_collection_store(
            vector_dimension=3, use_jsonb=True
        )
        self.EmbeddingStore = EmbeddingStore

    def get_all_ids(self) -> list[str]:
        return ["id1", "id2"]


def test_extended_pgvector_get_all_ids():
    dummy_vector = DummyPgVector()
    ids = dummy_vector.get_all_ids()
    assert ids == ["id1", "id2"]


def _compile(clause):
    return str(
        clause.compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )


class TestHandleFieldFilter:
    """Verify ExtendedPgVector emits index-friendly SQL for $eq/$ne filters."""

    @pytest.fixture
    def store(self):
        return DummyPgVector()

    def test_eq_uses_astext_not_jsonb_path_match(self, store):
        clause = store._handle_field_filter("file_id", {"$eq": "abc-123"})
        sql = _compile(clause)
        assert "->>" in sql, f"Expected ->> operator, got: {sql}"
        assert "jsonb_path_match" not in sql, f"Should not use jsonb_path_match: {sql}"
        assert "'abc-123'" in sql

    def test_ne_uses_astext_not_jsonb_path_match(self, store):
        clause = store._handle_field_filter("file_id", {"$ne": "abc-123"})
        sql = _compile(clause)
        assert "->>" in sql
        assert "jsonb_path_match" not in sql
        assert "!=" in sql or "<>" in sql

    def test_bare_value_treated_as_eq(self, store):
        clause = store._handle_field_filter("file_id", "abc-123")
        sql = _compile(clause)
        assert "->>" in sql
        assert "jsonb_path_match" not in sql

    def test_in_still_uses_astext(self, store):
        clause = store._handle_field_filter("file_id", {"$in": ["a", "b"]})
        sql = _compile(clause)
        assert "->>" in sql
        assert "IN" in sql

    def test_gt_delegates_to_parent(self, store):
        clause = store._handle_field_filter("score", {"$gt": 5})
        sql = _compile(clause)
        assert "jsonb_path_match" in sql

    def test_invalid_field_raises(self, store):
        with pytest.raises(ValueError, match="operator"):
            store._handle_field_filter("$bad", "val")

    def test_invalid_operator_raises(self, store):
        with pytest.raises(ValueError, match="Invalid operator"):
            store._handle_field_filter("field", {"$bogus": "val"})
