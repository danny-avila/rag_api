import os
import jwt
import pytest
from app.middleware import security_middleware

# Dummy Request class for testing.
class DummyRequest:
    def __init__(self, path, headers):
        self.url = type("URL", (), {"path": path})
        self.headers = headers
        self.state = type("State", (), {})()

async def dummy_call_next(request):
    return type("DummyResponse", (), {"status_code": 200})()

@pytest.fixture
def valid_jwt_header():
    jwt_secret = "testsecret"
    os.environ["JWT_SECRET"] = jwt_secret
    payload = {"id": "testuser", "exp": 9999999999}
    token = jwt.encode(payload, jwt_secret, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture
def invalid_jwt_header():
    return {"Authorization": "Bearer invalidtoken"}

@pytest.mark.asyncio
async def test_security_middleware_valid(valid_jwt_header):
    request = DummyRequest("/protected", valid_jwt_header)
    response = await security_middleware(request, dummy_call_next)
    assert response.status_code == 200
    assert hasattr(request.state, "user")
    assert request.state.user["id"] == "testuser"

@pytest.mark.asyncio
async def test_security_middleware_invalid(invalid_jwt_header):
    request = DummyRequest("/protected", invalid_jwt_header)
    response = await security_middleware(request, dummy_call_next)
    assert response.status_code == 401