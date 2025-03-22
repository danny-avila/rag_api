import pytest
from config import get_env_variable

def test_get_env_variable_with_value(monkeypatch):
    monkeypatch.setenv("TEST_VAR", "test_value")
    value = get_env_variable("TEST_VAR")
    assert value == "test_value"

def test_get_env_variable_with_default(monkeypatch):
    monkeypatch.delenv("NON_EXISTENT_VAR", raising=False)
    value = get_env_variable("NON_EXISTENT_VAR", default_value="default")
    assert value == "default"

def test_get_env_variable_required(monkeypatch):
    monkeypatch.delenv("NON_EXISTENT_REQUIRED", raising=False)
    with pytest.raises(ValueError):
        get_env_variable("NON_EXISTENT_REQUIRED", required=True)