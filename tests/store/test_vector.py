import pytest
from app.store.vector import (
    get_vector_store,
    ExtendedPgVector,
    AsyncPgVector,
    AtlasMongoVector,
)
from langchain_core.embeddings import Embeddings


# Dummy embeddings' implementation.
class DummyEmbeddings(Embeddings):
    def embed_query(self, query: str):
        return [0.1, 0.2, 0.3]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


# Patch the create_vector_extension method to do nothing for tests using SQLite.
@pytest.fixture(autouse=True)
def patch_vector_extension(monkeypatch):
    monkeypatch.setattr(ExtendedPgVector, "create_vector_extension", lambda self: None)


def test_get_vector_store_sync():
    vs = get_vector_store(
        connection_string="sqlite:///:memory:",
        embeddings=DummyEmbeddings(),
        collection_name="dummy_collection",
        mode="sync",
    )
    # Ensure that we get an instance of ExtendedPgVector.
    assert isinstance(vs, ExtendedPgVector)


def test_get_vector_store_async():
    vs = get_vector_store(
        connection_string="sqlite:///:memory:",
        embeddings=DummyEmbeddings(),
        collection_name="dummy_collection",
        mode="async",
    )
    # Ensure that we get an instance of AsyncPgVector.
    assert isinstance(vs, AsyncPgVector)


# --- Atlas Mongo Tests ---
# Create dummy classes to simulate a MongoDB connection.
def find(query):
    # Return a list of dummy document dictionaries.
    return [
        {
            "text": "dummy text",
            "file_id": "dummy_id1",
            "user_id": "public",
            "digest": "abc123",
            "source": "dummy_source",
            "page": 1,
        }
    ]


class DummyCollection:
    def distinct(self, field):
        return ["dummy_id1", "dummy_id2"]

    def delete_many(self, query):
        pass


class DummyDatabase:
    def __getitem__(self, collection_name):
        return DummyCollection()


class DummyMongoClient:
    def __init__(self, connection_string):
        self.connection_string = connection_string

    def get_database(self):
        return DummyDatabase()


# Patch pymongo.MongoClient so that get_vector_store uses our dummy.
@pytest.fixture(autouse=True)
def patch_mongo_client(monkeypatch):
    monkeypatch.setattr("pymongo.MongoClient", DummyMongoClient)


def test_get_vector_store_atlas_mongo():
    vs = get_vector_store(
        connection_string="dummy_conn",
        embeddings=DummyEmbeddings(),
        collection_name="dummy_collection",
        mode="atlas-mongo",
        search_index="dummy_index",
    )
    # Ensure that we get an instance of AtlasMongoVector.
    assert isinstance(vs, AtlasMongoVector)
    # Test that get_all_ids returns our dummy IDs.
    ids = vs.get_all_ids()
    assert isinstance(ids, list)
    assert "dummy_id1" in ids
    assert "dummy_id2" in ids