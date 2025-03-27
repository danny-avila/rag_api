import pytest
from app.utils.health import is_health_ok

def test_health_ok(monkeypatch):
    monkeypatch.setattr("app.utils.health.pg_health_check", lambda: True)
    monkeypatch.setattr("app.utils.health.mongo_health_check", lambda: True)
    result = is_health_ok()
    assert result is True