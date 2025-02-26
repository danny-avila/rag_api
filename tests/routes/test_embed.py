import os
import pytest
import jwt
import datetime
import importlib.util
from fastapi.testclient import TestClient

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


# Override the get_loader function to use a dummy loader.
@pytest.fixture(autouse=True)
def override_get_loader(monkeypatch):
    class DummyLoader:
        def load(self):
            from langchain.schema import Document
            return [Document(page_content="dummy content", metadata={"page": 1, "source": "dummy.txt"})]

    # Override get_loader in app.routes.embed.
    monkeypatch.setattr(
        "app.routes.embed.get_loader",
        lambda filename, content_type, filepath: (DummyLoader(), True, "txt")
    )

    # Check if the embed_local module exists; if so, override its get_loader as well.
    if importlib.util.find_spec("app.routes.embed_local") is not None:
        monkeypatch.setattr(
            "app.routes.embed_local.get_loader",
            lambda filename, content_type, filepath: (DummyLoader(), True, "txt")
        )


client = TestClient(app)


def test_embed_file_upload(tmp_path):
    # Create a temporary dummy file.
    dummy_content = b"dummy content"
    dummy_file = tmp_path / "dummy.txt"
    dummy_file.write_bytes(dummy_content)

    with open(dummy_file, "rb") as f:
        response = client.post(
            "/embed-upload",
            files={"uploaded_file": ("dummy.txt", f, "text/plain")},
            data={"file_id": "dummy_file", "entity_id": "public"},
            headers=get_auth_header()
        )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] is True
    assert data["file_id"] == "dummy_file"
    assert "dummy.txt" in data["filename"]