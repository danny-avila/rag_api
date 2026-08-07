"""Fail-closed auth for the service endpoints.

The load-bearing property is key separation: LibreChat's application
``JWT_SECRET`` signs session tokens, and a session token must never be usable
against ``/v1/*``. ``RAG_AUTH_ACCEPT_LEGACY`` relaxes the *claim shape* during
the transition; it never relaxes which key is trusted on those endpoints.
"""

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app import auth
from app.services import space as space_module
from main import app
from tests.fakes import FakeEmbeddingClient, install_fake_space
from tests.tokens import (
    APP_SECRET,
    AUDIENCE,
    ISSUER,
    RAG_SECRET,
    SYSTEM_TENANT,
    bearer,
    legacy_token,
    strict_token,
)

client = TestClient(app)

EMBED_BODY = {
    "space": space_module.CHAT_SPACE_SPEC.name,
    "input_type": "query",
    "inputs": [{"id": "a", "text": "hello world"}],
}

QUERY_BODY = {"query": "hello", "file_id": "file-1", "k": 4}


@pytest.fixture
def search_enabled(monkeypatch):
    monkeypatch.setenv("RAG_SEARCH_API_ENABLED", "true")
    monkeypatch.setenv("RAG_JWT_SECRET", RAG_SECRET)
    monkeypatch.setenv("JWT_SECRET", APP_SECRET)
    monkeypatch.setenv("RAG_JWT_ISSUER", ISSUER)
    monkeypatch.setenv("RAG_JWT_AUDIENCE", AUDIENCE)
    monkeypatch.setenv("RAG_AUTH_ACCEPT_LEGACY", "true")
    monkeypatch.setenv("RAG_RATE_LIMIT_ENABLED", "false")
    auth.reset_settings()
    install_fake_space(monkeypatch, FakeEmbeddingClient())
    if getattr(app.state, "thread_pool", None) is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)


@pytest.fixture
def store_stub(monkeypatch):
    monkeypatch.setattr(
        "app.services.embedding.get_cached_query_embedding",
        lambda query: [0.1, 0.2, 0.3],
    )
    monkeypatch.setattr(
        "app.routes.document_routes.get_cached_query_embedding",
        lambda query: [0.1, 0.2, 0.3],
    )

    async def asearch(self, embedding, k, filter=None, executor=None):
        return []

    from app.services.vector_store.async_pg_vector import AsyncPgVector

    monkeypatch.setattr(
        AsyncPgVector, "asimilarity_search_with_score_by_vector", asearch
    )
    if getattr(app.state, "thread_pool", None) is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)


def post_embed(token: str):
    return client.post("/v1/embeddings", json=EMBED_BODY, headers=bearer(token))


def test_strict_token_is_accepted(search_enabled):
    assert post_embed(strict_token()).status_code == 200


def test_librechat_session_token_is_rejected_by_the_service_endpoints(search_enabled):
    """The gate: a captured session token must not reach /v1/embeddings."""
    response = post_embed(legacy_token(user_id="user-1", secret=APP_SECRET))
    assert response.status_code == 401


def test_session_token_is_rejected_even_with_full_claims(search_enabled):
    """Claim shape is not the control — the signing key is."""
    token = strict_token(secret=APP_SECRET)
    assert post_embed(token).status_code == 401


def test_legacy_shape_signed_with_the_rag_key_is_accepted_during_transition(
    search_enabled,
):
    token = legacy_token(user_id="user-1", secret=RAG_SECRET)
    assert post_embed(token).status_code == 200


def test_legacy_shape_is_rejected_once_the_transition_flag_flips(
    search_enabled, monkeypatch
):
    monkeypatch.setenv("RAG_AUTH_ACCEPT_LEGACY", "false")
    auth.reset_settings()
    token = legacy_token(user_id="user-1", secret=RAG_SECRET)
    assert post_embed(token).status_code == 401
    assert post_embed(strict_token()).status_code == 200


def test_missing_scope_is_forbidden(search_enabled):
    response = post_embed(strict_token(scopes=["rag:rerank"]))
    assert response.status_code == 403
    assert "rag:embed" in response.json()["detail"]


def test_rerank_requires_its_own_scope(search_enabled):
    body = {
        "profile": "fast-v1",
        "query": "q",
        "candidates": [{"id": "c1", "text": "t", "base_score": 1.0}],
    }
    response = client.post(
        "/v1/rerank", json=body, headers=bearer(strict_token(scopes=["rag:embed"]))
    )
    assert response.status_code == 403


def test_wrong_audience_is_rejected(search_enabled):
    assert post_embed(strict_token(audience="librechat")).status_code == 401


def test_wrong_issuer_is_rejected(search_enabled):
    assert post_embed(strict_token(issuer="somebody-else")).status_code == 401


def test_missing_audience_falls_back_to_legacy_only_while_permitted(
    search_enabled, monkeypatch
):
    token = strict_token(audience=None)
    assert post_embed(token).status_code == 200
    monkeypatch.setenv("RAG_AUTH_ACCEPT_LEGACY", "false")
    auth.reset_settings()
    assert post_embed(strict_token(audience=None)).status_code == 401


def test_missing_tenant_claim_is_rejected_under_strict_acceptance(
    search_enabled, monkeypatch
):
    monkeypatch.setenv("RAG_AUTH_ACCEPT_LEGACY", "false")
    auth.reset_settings()
    assert post_embed(strict_token(tenant=None)).status_code == 401


def test_missing_scopes_claim_is_rejected_under_strict_acceptance(
    search_enabled, monkeypatch
):
    monkeypatch.setenv("RAG_AUTH_ACCEPT_LEGACY", "false")
    auth.reset_settings()
    assert post_embed(strict_token(scopes=None)).status_code == 401


def test_system_tenant_is_forbidden(search_enabled):
    response = post_embed(strict_token(tenant=SYSTEM_TENANT))
    assert response.status_code == 403
    assert "System tenant" in response.json()["detail"]


def test_expired_token_is_rejected(search_enabled):
    assert post_embed(strict_token(expires_in=-60)).status_code == 401


def test_token_without_expiry_is_rejected_under_strict_acceptance(
    search_enabled, monkeypatch
):
    import jwt as pyjwt

    monkeypatch.setenv("RAG_AUTH_ACCEPT_LEGACY", "false")
    auth.reset_settings()
    token = pyjwt.encode(
        {
            "sub": "user-1",
            "iss": ISSUER,
            "aud": AUDIENCE,
            "tenant": "__BASE__",
            "scopes": ["rag:embed"],
        },
        RAG_SECRET,
        algorithm="HS256",
    )
    assert post_embed(token).status_code == 401


def test_missing_authorization_header_is_rejected(search_enabled):
    assert client.post("/v1/embeddings", json=EMBED_BODY).status_code == 401


def test_service_endpoints_are_unavailable_when_the_search_api_is_off(monkeypatch):
    monkeypatch.setenv("RAG_SEARCH_API_ENABLED", "false")
    monkeypatch.setenv("RAG_JWT_SECRET", RAG_SECRET)
    monkeypatch.setenv("JWT_SECRET", APP_SECRET)
    auth.reset_settings()
    assert post_embed(strict_token()).status_code == 503


def test_legacy_routes_still_accept_the_application_secret(search_enabled, store_stub):
    response = client.post(
        "/query", json=QUERY_BODY, headers=bearer(legacy_token(secret=APP_SECRET))
    )
    assert response.status_code == 200


def test_legacy_routes_reject_the_application_secret_once_the_flag_flips(
    search_enabled, store_stub, monkeypatch
):
    monkeypatch.setenv("RAG_AUTH_ACCEPT_LEGACY", "false")
    auth.reset_settings()
    response = client.post(
        "/query", json=QUERY_BODY, headers=bearer(legacy_token(secret=APP_SECRET))
    )
    assert response.status_code == 401
    assert (
        client.post(
            "/query", json=QUERY_BODY, headers=bearer(strict_token())
        ).status_code
        == 200
    )


class TestStartupValidation:
    def _settings(self, monkeypatch, **env):
        for key, value in env.items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)
        auth.reset_settings()
        return auth.get_settings()

    def test_valid_configuration_passes(self, monkeypatch):
        settings = self._settings(
            monkeypatch,
            RAG_SEARCH_API_ENABLED="true",
            RAG_JWT_SECRET=RAG_SECRET,
            JWT_SECRET=APP_SECRET,
        )
        settings.validate()

    def test_shared_signing_key_is_refused(self, monkeypatch):
        settings = self._settings(
            monkeypatch,
            RAG_SEARCH_API_ENABLED="true",
            RAG_JWT_SECRET=APP_SECRET,
            JWT_SECRET=APP_SECRET,
        )
        with pytest.raises(RuntimeError, match="must differ from JWT_SECRET"):
            settings.validate()

    def test_enabled_without_a_signing_key_is_refused(self, monkeypatch):
        settings = self._settings(
            monkeypatch,
            RAG_SEARCH_API_ENABLED="true",
            RAG_JWT_SECRET=None,
            JWT_SECRET=APP_SECRET,
        )
        with pytest.raises(RuntimeError, match="requires RAG_JWT_SECRET"):
            settings.validate()

    def test_short_hmac_secret_is_refused(self, monkeypatch):
        settings = self._settings(
            monkeypatch,
            RAG_SEARCH_API_ENABLED="true",
            RAG_JWT_SECRET="too-short",
            JWT_SECRET=APP_SECRET,
        )
        with pytest.raises(RuntimeError, match="at least 32 characters"):
            settings.validate()

    def test_unsupported_algorithm_is_refused(self, monkeypatch):
        settings = self._settings(
            monkeypatch,
            RAG_SEARCH_API_ENABLED="true",
            RAG_JWT_SECRET=RAG_SECRET,
            RAG_JWT_ALGORITHM="none",
        )
        with pytest.raises(RuntimeError, match="not supported"):
            settings.validate()

    def test_strict_acceptance_without_a_key_is_refused(self, monkeypatch):
        settings = self._settings(
            monkeypatch,
            RAG_SEARCH_API_ENABLED="false",
            RAG_AUTH_ACCEPT_LEGACY="false",
            RAG_JWT_SECRET=None,
            JWT_SECRET=APP_SECRET,
        )
        with pytest.raises(RuntimeError, match="requires a verification key"):
            settings.validate()

    def test_asymmetric_configuration_requires_a_public_key(self, monkeypatch):
        settings = self._settings(
            monkeypatch,
            RAG_SEARCH_API_ENABLED="true",
            RAG_JWT_ALGORITHM="RS256",
            RAG_JWT_SECRET=RAG_SECRET,
            RAG_JWT_PUBLIC_KEY=None,
        )
        with pytest.raises(RuntimeError, match="requires RAG_JWT_PUBLIC_KEY"):
            settings.validate()
