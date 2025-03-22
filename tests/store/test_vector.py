import pytest
from store_factory import get_vector_store
from langchain_core.embeddings import Embeddings

# Dummy embeddings implementation.
class DummyEmbeddings(Embeddings):
    def embed_query(self, query: str):
        return [0.1, 0.2, 0.3]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


def test_get_vector_store_dummy_sync():
    vs = get_vector_store(
        connection_string="dummy_conn",
        embeddings=DummyEmbeddings(),
        collection_name="dummy_collection",
        mode="dummy",
    )
    # In dummy mode, get_all_ids should return an empty list.
    assert vs.get_all_ids() == []
    # Similarly, get_documents_by_ids should return an empty list.
    assert vs.get_documents_by_ids(["id1", "id2"]) == []
    # delete should be callable without raising an error.
    vs.delete(ids=["id1", "id2"], collection_only=True)


@pytest.mark.asyncio
async def test_get_vector_store_dummy_async():
    vs = get_vector_store(
        connection_string="dummy_conn",
        embeddings=DummyEmbeddings(),
        collection_name="dummy_collection",
        mode="dummy",
    )
    # Even for async, since dummy mode doesn't require async behavior,
    # the same interface applies.
    assert vs.get_all_ids() == []
    assert vs.get_documents_by_ids(["id1", "id2"]) == []
    vs.delete(ids=["id1", "id2"], collection_only=True)


# --- Atlas Mongo Tests in Dummy Mode ---
def test_get_vector_store_dummy_atlas_mongo():
    vs = get_vector_store(
        connection_string="dummy_conn",
        embeddings=DummyEmbeddings(),
        collection_name="dummy_collection",
        mode="dummy",
        search_index="dummy_index",
    )
    # In dummy mode, this should also return an empty list of IDs.
    assert vs.get_all_ids() == []