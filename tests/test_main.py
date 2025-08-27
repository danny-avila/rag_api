import os
import jwt
import datetime
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor

from main import app

client = TestClient(app)


@pytest.fixture
def auth_headers():
    jwt_secret = "testsecret"
    os.environ["JWT_SECRET"] = jwt_secret
    payload = {
        "id": "testuser",
        "exp": datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(hours=1),
    }
    token = jwt.encode(payload, jwt_secret, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def override_vector_store(monkeypatch):
    from app.config import vector_store

    # Initialize thread pool for tests since TestClient doesn't run lifespan
    if not hasattr(app.state, "thread_pool") or app.state.thread_pool is None:
        app.state.thread_pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="test-worker"
        )

    # Override get_all_ids as an async function.
    async def dummy_get_all_ids(executor=None):
        return ["testid1", "testid2"]

    monkeypatch.setattr(vector_store, "get_all_ids", dummy_get_all_ids)

    # Override get_filtered_ids as an async function.
    async def dummy_get_filtered_ids(ids, executor=None):
        dummy_ids = ["testid1", "testid2"]
        return [id for id in dummy_ids if id in ids]

    monkeypatch.setattr(vector_store, "get_filtered_ids", dummy_get_filtered_ids)

    # Override get_documents_by_ids as an async function.
    async def dummy_get_documents_by_ids(ids, executor=None):
        return [
            Document(page_content="Test content", metadata={"file_id": id})
            for id in ids
        ]

    monkeypatch.setattr(
        vector_store, "get_documents_by_ids", dummy_get_documents_by_ids
    )

    # Override embedding_function.
    class DummyEmbedding:
        def embed_query(self, query):
            return [0.1, 0.2, 0.3]

    vector_store.embedding_function = DummyEmbedding()

    # Override similarity search to return a tuple (Document, score).
    def dummy_similarity_search_with_score_by_vector(embedding, k, filter):
        doc = Document(
            page_content="Queried content",
            metadata={
                "file_id": filter.get("file_id", "testid1"),
                "user_id": "testuser",
            },
        )
        return [(doc, 0.9)]

    async def dummy_asimilarity_search_with_score_by_vector(
        embedding, k, filter=None, executor=None
    ):
        doc = Document(
            page_content="Queried content",
            metadata={
                "file_id": filter.get("file_id", "testid1") if filter else "testid1",
                "user_id": "testuser",
            },
        )
        return [(doc, 0.9)]

    monkeypatch.setattr(
        vector_store,
        "similarity_search_with_score_by_vector",
        dummy_similarity_search_with_score_by_vector,
    )
    monkeypatch.setattr(
        vector_store,
        "asimilarity_search_with_score_by_vector",
        dummy_asimilarity_search_with_score_by_vector,
    )

    # Override document addition functions.
    def dummy_add_documents(docs, ids):
        return ids

    async def dummy_aadd_documents(docs, ids=None, executor=None):
        return ids

    monkeypatch.setattr(vector_store, "add_documents", dummy_add_documents)
    monkeypatch.setattr(vector_store, "aadd_documents", dummy_aadd_documents)

    # Override delete function.
    async def dummy_delete(ids=None, collection_only=False, executor=None):
        return None

    monkeypatch.setattr(vector_store, "delete", dummy_delete)


def test_get_all_ids(auth_headers):
    response = client.get("/ids", headers=auth_headers)
    assert response.status_code == 200
    json_data = response.json()
    assert isinstance(json_data, list)
    assert "testid1" in json_data


def test_get_documents_by_ids(auth_headers):
    response = client.get(
        "/documents", params={"ids": ["testid1"]}, headers=auth_headers
    )
    assert response.status_code == 200
    json_data = response.json()
    assert isinstance(json_data, list)
    assert json_data[0]["page_content"] == "Test content"
    assert json_data[0]["metadata"]["file_id"] == "testid1"


def test_delete_documents(auth_headers):
    response = client.request(
        "DELETE", "/documents", json=["testid1"], headers=auth_headers
    )
    assert response.status_code == 200
    json_data = response.json()
    assert "Documents for" in json_data["message"]


def test_query_embeddings_by_file_id(auth_headers):
    data = {
        "query": "Test query",
        "file_id": "testid1",
        "k": 4,
        "entity_id": "testuser",
    }
    response = client.post("/query", json=data, headers=auth_headers)
    assert response.status_code == 200
    json_data = response.json()
    assert isinstance(json_data, list)
    if json_data:
        doc = json_data[0][0]
        assert doc["page_content"] == "Queried content"


def test_embed_local_file(tmp_path, auth_headers, monkeypatch):
    # Create a temporary file.
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test document.")

    data = {
        "filepath": str(test_file),
        "filename": "test.txt",
        "file_content_type": "text/plain",
        "file_id": "testid1",
    }
    response = client.post("/local/embed", json=data, headers=auth_headers)
    assert response.status_code == 200, f"Response: {response.text}"
    json_data = response.json()
    assert json_data["status"] is True
    assert json_data["file_id"] == "testid1"


def test_embed_file(tmp_path, auth_headers):
    file_content = "This is a test file for the embed endpoint."
    test_file = tmp_path / "test_embed.txt"
    test_file.write_text(file_content)
    with test_file.open("rb") as f:
        response = client.post(
            "/embed",
            data={"file_id": "testid1", "entity_id": "testuser"},
            files={"file": ("test_embed.txt", f, "text/plain")},
            headers=auth_headers,
        )
    assert response.status_code == 200, f"Response: {response.text}"
    json_data = response.json()
    assert json_data["status"] is True
    assert json_data["file_id"] == "testid1"


def test_load_document_context(auth_headers):
    response = client.get("/documents/testid1/context", headers=auth_headers)
    assert response.status_code == 200, f"Response: {response.text}"
    content = response.text
    assert "testid1" in content or "Test content" in content


def test_embed_file_upload(tmp_path, auth_headers, monkeypatch):
    file_content = "Test content for embed upload."
    test_file = tmp_path / "upload_test.txt"
    test_file.write_text(file_content)

    with test_file.open("rb") as f:
        response = client.post(
            "/embed-upload",
            data={"file_id": "testid1", "entity_id": "testuser"},
            files={"uploaded_file": ("upload_test.txt", f, "text/plain")},
            headers=auth_headers,
        )
    assert response.status_code == 200, f"Response: {response.text}"
    json_data = response.json()
    assert json_data["status"] is True
    assert json_data["file_id"] == "testid1"


def test_query_multiple(auth_headers):
    data = {
        "query": "Test query multiple",
        "file_ids": ["testid1", "testid2"],
        "k": 4,
    }
    response = client.post("/query_multiple", json=data, headers=auth_headers)
    assert response.status_code == 200, f"Response: {response.text}"
    json_data = response.json()
    assert isinstance(json_data, list)
    if json_data:
        doc = json_data[0][0]
        assert doc["page_content"] == "Queried content"


def test_extract_text_from_file(tmp_path, auth_headers):
    """Test the /text endpoint for text extraction without embeddings."""
    file_content = "This is a test file for text extraction.\nIt has multiple lines.\nAnd should be extracted properly."
    test_file = tmp_path / "test_text_extraction.txt"
    test_file.write_text(file_content)

    with test_file.open("rb") as f:
        response = client.post(
            "/text",
            data={"file_id": "test_text_123", "entity_id": "testuser"},
            files={"file": ("test_text_extraction.txt", f, "text/plain")},
            headers=auth_headers,
        )

    assert response.status_code == 200, f"Response: {response.text}"
    json_data = response.json()

    # Check response structure
    assert "text" in json_data
    assert "file_id" in json_data
    assert "filename" in json_data
    assert "known_type" in json_data

    # Check response content
    assert json_data["text"] == file_content
    assert json_data["file_id"] == "test_text_123"
    assert json_data["filename"] == "test_text_extraction.txt"
    assert json_data["known_type"] is True  # text files are known types
