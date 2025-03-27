import jwt
import datetime
from fastapi.testclient import TestClient
from langchain.schema import Document
from config import get_env_variable
from main import app

# Dummy embeddings for query endpoints.
class DummyEmbeddings:
    def embed_query(self, query: str):
        return [0.1, 0.2, 0.3]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

# Dummy vector store for route testing.
class DummyVectorStore:
    # Provide a dummy embedding function.
    embedding_function = DummyEmbeddings()

    def get_all_ids(self):
        return ["dummy_id"]

    def get_documents_by_ids(self, ids):
        if "dummy_id" in ids:
            return [Document(page_content="dummy content", metadata={"file_id": "dummy_id", "user_id": "public"})]
        return []

    def delete(self, ids, collection_only=False):
        # Simulate successful deletion (but leave get_all_ids unchanged).
        pass

    def similarity_search_with_score_by_vector(self, embedding, k, filter):
        # Return a dummy result for queries filtering by file_id "test".
        if filter.get("file_id") == "test":
            doc = Document(page_content="dummy content", metadata={"file_id": "test", "user_id": "public"})
            return [(doc, 0.9)]
        file_id_filter = filter.get("file_id")
        if isinstance(file_id_filter, dict) and "$in" in file_id_filter:
            if "test" in file_id_filter["$in"]:
                doc = Document(page_content="dummy content", metadata={"file_id": "test", "user_id": "public"})
                return [(doc, 0.9)]
        return []

import pytest

@pytest.fixture(autouse=True)
def override_vector_store(monkeypatch):
    dummy = DummyVectorStore()
    # Override the vector_store attribute in config.
    monkeypatch.setattr("config.vector_store", dummy)

client = TestClient(app)