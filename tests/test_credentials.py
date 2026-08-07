"""No hardcoded credentials anywhere in the tracked configuration surface.

A working fallback default is the defect, not the convenience: a sample
user/password pair in a compose file is how a database ends up reachable with
credentials that are public in the repository.
"""

import importlib
import os
import subprocess
from pathlib import Path

import pytest

from app import auth
from app.config import require_env_variable

REPO_ROOT = Path(__file__).resolve().parents[1]

# Values that used to ship as working defaults.
BANNED_LITERALS = ("mypassword", "myuser", "mydatabase")

CONFIG_FILES = (
    "app/config.py",
    "docker-compose.yaml",
    "db-compose.yaml",
    "api-compose.yaml",
    "README.md",
    ".env.example",
)


def _tracked_files() -> list:
    result = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return [line for line in result.stdout.splitlines() if line]


class TestNoHardcodedCredentials:
    @pytest.mark.parametrize("relative_path", CONFIG_FILES)
    def test_config_file_carries_no_default_credential(self, relative_path):
        content = (REPO_ROOT / relative_path).read_text()
        for literal in BANNED_LITERALS:
            assert literal not in content, f"{relative_path} still ships '{literal}'"

    def test_no_tracked_file_ships_a_default_credential(self):
        offenders = []
        for relative_path in _tracked_files():
            # This file names the banned literals in order to search for them.
            if relative_path == "tests/test_credentials.py":
                continue
            path = REPO_ROOT / relative_path
            if not path.is_file() or path.suffix in (".png", ".jpg", ".ico"):
                continue
            try:
                content = path.read_text()
            except UnicodeDecodeError:
                continue
            if any(literal in content for literal in BANNED_LITERALS):
                offenders.append(relative_path)
        assert offenders == []

    def test_compose_files_require_their_credentials(self):
        for relative_path in ("docker-compose.yaml", "db-compose.yaml"):
            content = (REPO_ROOT / relative_path).read_text()
            for variable in ("POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD"):
                assert (
                    f"${{{variable}:?" in content
                ), f"{relative_path} does not require {variable}"

    def test_env_example_is_tracked_and_only_holds_placeholders(self):
        assert ".env.example" in _tracked_files()
        assert ".env" not in _tracked_files()
        for line in (REPO_ROOT / ".env.example").read_text().splitlines():
            if not line or line.startswith("#") or "=" not in line:
                continue
            _, _, value = line.partition("=")
            assert value.startswith("REPLACE_ME_") or value in ("db", "5432"), line


class TestRequiredEnvVariable:
    def test_a_missing_variable_is_refused(self, monkeypatch):
        monkeypatch.delenv("SOME_CREDENTIAL", raising=False)
        with pytest.raises(ValueError, match="required and has no default"):
            require_env_variable("SOME_CREDENTIAL")

    def test_an_empty_variable_is_refused(self, monkeypatch):
        monkeypatch.setenv("SOME_CREDENTIAL", "")
        with pytest.raises(ValueError, match="required and has no default"):
            require_env_variable("SOME_CREDENTIAL")

    def test_an_empty_variable_is_allowed_only_when_asked_for(self, monkeypatch):
        monkeypatch.setenv("SOME_CREDENTIAL", "")
        assert require_env_variable("SOME_CREDENTIAL", allow_empty=True) == ""

    def test_a_present_variable_is_returned(self, monkeypatch):
        monkeypatch.setenv("SOME_CREDENTIAL", "value")
        assert require_env_variable("SOME_CREDENTIAL") == "value"

    def test_the_postgres_credentials_have_no_fallback(self, monkeypatch):
        """Re-importing app.config without credentials must fail, not default."""
        monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
        import app.config

        with pytest.raises(ValueError, match="POSTGRES_PASSWORD"):
            importlib.reload(app.config)
        # Restore the module for the rest of the session.
        monkeypatch.setenv("POSTGRES_PASSWORD", "rag_api_test_password")
        importlib.reload(app.config)


class TestSigningConfigHasNoFallback:
    def _settings(self, monkeypatch, **env):
        for key, value in env.items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)
        auth.reset_settings()
        return auth.get_settings()

    def test_startup_fails_without_a_signing_key(self, monkeypatch):
        self._settings(
            monkeypatch,
            RAG_SEARCH_API_ENABLED="true",
            RAG_JWT_SECRET=None,
            RAG_JWT_PUBLIC_KEY=None,
        )
        with pytest.raises(RuntimeError, match="requires RAG_JWT_SECRET"):
            auth.validate_startup_config()

    def test_there_is_no_generated_or_default_signing_key(self, monkeypatch):
        settings = self._settings(
            monkeypatch,
            RAG_SEARCH_API_ENABLED="false",
            RAG_JWT_SECRET=None,
            RAG_JWT_PUBLIC_KEY=None,
        )
        assert settings.rag_secret is None
        assert settings.verification_key is None

    def test_secrets_never_appear_in_the_startup_log(self, monkeypatch, caplog):
        import logging

        secret = "a-very-secret-signing-key-0123456789"
        self._settings(
            monkeypatch,
            RAG_SEARCH_API_ENABLED="true",
            RAG_JWT_SECRET=secret,
            JWT_SECRET="a-different-application-secret-0123456789",
        )
        with caplog.at_level(logging.DEBUG):
            auth.validate_startup_config()
        assert secret not in caplog.text

    def test_a_rejected_token_never_echoes_a_secret(self, monkeypatch):
        secret = "a-very-secret-signing-key-0123456789"
        settings = self._settings(
            monkeypatch,
            RAG_SEARCH_API_ENABLED="true",
            RAG_JWT_SECRET=secret,
            JWT_SECRET="a-different-application-secret-0123456789",
            RAG_AUTH_ACCEPT_LEGACY="false",
        )
        with pytest.raises(auth.AuthError) as excinfo:
            auth.verify_token("not-a-token", settings, allow_legacy_secret=False)
        assert secret not in str(excinfo.value)
