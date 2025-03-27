import pytest
import jwt
import datetime
from fastapi.testclient import TestClient
from langchain.schema import Document
from config import get_env_variable
from main import app

JWT_SECRET = get_env_variable("JWT_SECRET", required=True)

def get_auth_header():
    token = jwt.encode(
        {"id": "public", "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)},
        JWT_SECRET,
        algorithm="HS256",
    )
    return {"Authorization": f"Bearer {token}"}

class DummyEmbeddings:
    @staticmethod
    def embed_query(query: str):
        return [1.0, 2.0, 3.0]

class DummyVectorStore:
    embedding_function = DummyEmbeddings()

    @staticmethod
    def similarity_search_with_score_by_vector(embedding, k, filter):
        if filter.get("file_id") == "test":
            doc = Document(
                page_content="dummy content",
                metadata={"file_id": "test", "user_id": "public"}
            )
            return [(doc, 0.9)]
        file_id_filter = filter.get("file_id")
        if isinstance(file_id_filter, dict) and "$in" in file_id_filter:
            if "test" in file_id_filter["$in"]:
                doc = Document(
                    page_content="dummy content",
                    metadata={"file_id": "test", "user_id": "public"}
                )
                return [(doc, 0.9)]
        return []

@pytest.fixture(autouse=True)
def override_vector_store(monkeypatch):
    dummy = DummyVectorStore()
    # Override the vector_store in config.
    monkeypatch.setattr("config.vector_store", dummy)

client = TestClient(app)

def test_query_with_auth():
    payload = {"query": "dummy query", "file_id": "test", "k": 4}
    response = client.post("/query", json=payload, headers=get_auth_header())
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0

def test_query_multiple_with_auth():
    payload = {"query": "dummy query", "file_ids": ["test"], "k": 4}
    response = client.post("/query_multiple", json=payload, headers=get_auth_header())
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0