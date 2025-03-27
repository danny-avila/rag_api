from app.services.vector_store.extended_pg_vector import ExtendedPgVector

# Create a dummy subclass that simulates DB responses.
class DummyPgVector(ExtendedPgVector):
    def __init__(self):
        self._bind = None
        self.EmbeddingStore = None

    def get_all_ids(self) -> list[str]:
        return ["id1", "id2"]

def test_extended_pgvector_get_all_ids():
    dummy_vector = DummyPgVector()
    ids = dummy_vector.get_all_ids()
    assert ids == ["id1", "id2"]