from app.services.vector_store.extended_pg_vector import ExtendedPgVector
from app.services.vector_store.atlas_mongo_vector import AtlasMongoVector

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


def test_extended_pgvector_close_disposes_engine():
    class DummyBind:
        def __init__(self):
            self.disposed = False

        def dispose(self):
            self.disposed = True

    vec = DummyPgVector()
    vec._bind = DummyBind()
    vec.close()
    assert vec._bind.disposed is True


def test_atlas_mongo_vector_close_closes_client():
    class DummyClient:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class DummyDatabase:
        def __init__(self, client):
            self.client = client

    class DummyCollection:
        def __init__(self, client):
            self.database = DummyDatabase(client)

    client = DummyClient()
    collection = DummyCollection(client)

    # Bypass base class init; we only need _collection for close()
    vec = AtlasMongoVector.__new__(AtlasMongoVector)
    vec._collection = collection
    vec.close()
    assert client.closed is True