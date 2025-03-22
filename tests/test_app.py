from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    # Health endpoint may return 200 (UP) or 503 (DOWN) based on DB connectivity.
    assert response.status_code in [200, 503]
    data = response.json()
    # If the response is a list, then extract the first element (the actual body)
    if isinstance(data, list):
        data = data[0]
    assert "status" in data
    assert data["status"] in ["UP", "DOWN"]

def test_protected_route_without_auth():
    # Test an endpoint (e.g. /query) that requires JWT authorization.
    payload = {"query": "test", "file_id": "test", "k": 4}
    response = client.post("/query", json=payload)
    # Without a valid Authorization header, it should return 401.
    assert response.status_code == 401