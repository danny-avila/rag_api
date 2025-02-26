from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    # Health endpoint may return 200 (UP) or 503 (DOWN) based on DB connectivity.
    assert response.status_code in [200, 503]
    data = response.json()
    assert "status" in data
    assert data["status"] in ["UP", "DOWN"]