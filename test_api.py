from main import app
from fastapi.testclient import TestClient
import json
import time
import pytest

client = TestClient(app)


# GET /ids
def getIDs():
    response = client.get("/ids")
    assert response.status_code == 200
    return json.loads(response.text)


# DELETE /documents
def deleteID(id):
    response = client.request(method="DELETE", url="/documents", json=[id])
    assert response.status_code == 200


# POST /embed
def embed(filepath, file_name, file_id):
    with open(filepath, "rb") as file:
        file_data = file.read()
    response = client.post(
        url="/embed",
        files={"file": (file_name, file_data, "text/plain")},
        data={"file_id": file_id},
    )
    assert response.status_code == 200


# POST /query
def query(file_id, query, k=1):
    response = client.post(
        url="/query", json={"file_id": file_id, "query": query, "k": k}
    )
    assert response.status_code == 200
    docs = json.loads(response.text)
    return docs


# POST /query_multiple
def queryMultiple(file_ids: list[str], query, k):
    response = client.post(
        url="/query_multiple", json={"file_ids": file_ids, "query": query, "k": k}
    )
    assert response.status_code == 200
    docs = json.loads(response.text)
    return docs


@pytest.fixture(scope="session", autouse=True)
def setup_and_teardown():
    embed(
        filepath="testFiles/superbowl.txt",
        file_name="superbowl.txt",
        file_id="superbowl",
    )
    embed(
        filepath="testFiles/short.txt",
        file_name="short.txt",
        file_id="short",
    )
    time.sleep(5)  # Wait for database update
    yield
    deleteID("superbowl")
    ids = getIDs()
    assert "superbowl" not in ids
    assert "short" in ids
    deleteID("short")
    assert "short" not in getIDs()


def test_getIDs():
    ids = getIDs()
    assert "short" in ids
    assert "superbowl" in ids
    assert "superbowl_0" not in ids


def test_query():
    response = query("superbowl", "Who sang the national anthem in the 2024 superbowl?")
    assert "Reba McEntire" in response[0][0]["page_content"]
    response = query("superbowl", "what is LibreChat?")
    assert "LibreChat" not in response[0][0]["page_content"]


def test_queryMultiple():
    response = queryMultiple(["short", "superbowl"], "What is LibreChat?", 2)
    assert "short" == response[0][0]["metadata"]["file_id"]
    assert "superbowl" == response[1][0]["metadata"]["file_id"]
    response = queryMultiple(["superbowl"], "What is Librechat", 2)
    assert "superbowl" == response[0][0]["metadata"]["file_id"]
    assert "superbowl" == response[1][0]["metadata"]["file_id"]


