import pytest
import jwt
import datetime
from fastapi.testclient import TestClient
from langchain.schema import Document
from app.config import get_env_variable
from app.main import app

JWT_SECRET = get_env_variable("JWT_SECRET", required=True)

def get_auth_header():
    token = jwt.encode(
        {"id": "public", "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)},
        JWT_SECRET,
        algorithm="HS256",
    )
    return {"Authorization": f"Bearer {token}"}

# Dummy vector store for testing.
class DummyVectorStore:
    def get_all_ids(self):
        return ["dummy_id"]

    def get_documents_by_ids(self, ids):
        if "dummy_id" in ids:
            return [Document(page_content="dummy content", metadata={"file_id": "dummy_id", "user_id": "public"})]
        return []

    def delete(self, ids):
        # Dummy delete does nothing.
        pass

@pytest.fixture(autouse=True)
def override_vector_store(monkeypatch):
    dummy = DummyVectorStore()
    # Override where the vector store is imported.
    monkeypatch.setattr("app.config.vector_store", dummy)
    # Also override in the documents route module.
    monkeypatch.setattr("app.routes.documents.vector_store", dummy)

client = TestClient(app)

def test_get_all_ids():
    response = client.get("/ids", headers=get_auth_header())
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    for item in data:
        assert isinstance(item, str) and item.strip()

def test_get_documents_by_ids():
    response = client.get("/documents", params=[("ids", "dummy_id")], headers=get_auth_header())
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert data[0]["metadata"]["file_id"] == "dummy_id"

def test_delete_documents():
    response = client.request(
        "DELETE",
        "/documents",
        json=["dummy_id"],
        headers=get_auth_header()
    )
    assert response.status_code == 200
    data = response.json()
    assert "deleted successfully" in data["message"]