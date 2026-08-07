"""POST /v1/embeddings — contract, limits, and the chat-v1 substitution lock."""

import logging
import math
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app import auth
from app.constants import MAX_EMBEDDING_CHARS, MAX_EMBEDDING_INPUTS
from app.services import ratelimit
from app.services import space as space_module
from app.utils.text import content_hash, normalize_text
from main import app
from tests.fakes import (
    FAKE_MODEL,
    FakeEmbeddingClient,
    deterministic_vector,
    install_fake_space,
)
from tests.tokens import APP_SECRET, RAG_SECRET, bearer, legacy_token, strict_token

client = TestClient(app)

SPACE = space_module.CHAT_SPACE_SPEC.name


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setenv("RAG_SEARCH_API_ENABLED", "true")
    monkeypatch.setenv("RAG_JWT_SECRET", RAG_SECRET)
    monkeypatch.setenv("JWT_SECRET", APP_SECRET)
    monkeypatch.setenv("RAG_RATE_LIMIT_ENABLED", "false")
    auth.reset_settings()
    if getattr(app.state, "thread_pool", None) is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)
    fake = FakeEmbeddingClient()
    install_fake_space(monkeypatch, fake)
    return fake


def post(inputs, space=SPACE, input_type="query", token=None):
    return client.post(
        "/v1/embeddings",
        json={"space": space, "input_type": input_type, "inputs": inputs},
        headers=bearer(token or strict_token()),
    )


def test_response_shape_matches_the_contract(backend):
    response = post([{"id": "a", "text": "alpha"}, {"id": "b", "text": "beta"}])
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "space",
        "model",
        "dimensions",
        "normalized",
        "items",
        "usage",
    }
    assert payload["space"] == SPACE
    assert payload["model"] == FAKE_MODEL
    assert payload["dimensions"] == 8
    assert payload["normalized"] is True
    assert [item["id"] for item in payload["items"]] == ["a", "b"]
    assert set(payload["items"][0]) == {"id", "content_hash", "embedding"}
    assert payload["usage"] == {"input_count": 2, "total_characters": 9}


def test_caller_ids_and_order_are_preserved(backend):
    ids = ["z", "m", "a"]
    response = post([{"id": i, "text": f"text {i}"} for i in ids])
    assert [item["id"] for item in response.json()["items"]] == ids


def test_vectors_are_l2_normalized(backend):
    response = post([{"id": "a", "text": "alpha"}])
    vector = response.json()["items"][0]["embedding"]
    assert math.isclose(
        math.sqrt(sum(component**2 for component in vector)), 1.0, rel_tol=1e-9
    )


def test_content_hash_is_over_the_normalized_text(backend):
    response = post([{"id": "a", "text": "  Hello   world \n"}])
    item = response.json()["items"][0]
    assert item["content_hash"] == content_hash("Hello world")
    assert normalize_text("  Hello   world \n") == "Hello world"


def test_whitespace_variants_share_a_content_hash(backend):
    first = post([{"id": "a", "text": "one two"}]).json()["items"][0]
    second = post([{"id": "a", "text": "one\n\ttwo  "}]).json()["items"][0]
    assert first["content_hash"] == second["content_hash"]
    assert first["embedding"] == second["embedding"]


def test_input_limit_is_enforced(backend):
    inputs = [{"id": f"i{n}", "text": "x"} for n in range(MAX_EMBEDDING_INPUTS)]
    assert post(inputs).status_code == 200
    inputs.append({"id": "overflow", "text": "x"})
    assert post(inputs).status_code == 422


def test_aggregate_character_limit_is_enforced(backend):
    oversized = [
        {"id": "a", "text": "x" * (MAX_EMBEDDING_CHARS // 2)},
        {"id": "b", "text": "x" * (MAX_EMBEDDING_CHARS // 2 + 1)},
    ]
    assert post(oversized).status_code == 422


def test_duplicate_ids_are_rejected(backend):
    assert post([{"id": "a", "text": "x"}, {"id": "a", "text": "y"}]).status_code == 422


def test_empty_and_whitespace_only_text_is_rejected(backend):
    assert post([{"id": "a", "text": ""}]).status_code == 422
    assert post([{"id": "a", "text": "   \n "}]).status_code == 422


def test_input_type_is_constrained(backend):
    assert post([{"id": "a", "text": "x"}], input_type="passage").status_code == 422


class TestNormalizedSizeLimit:
    """The advertised limit is a provider limit, so it binds the sent text.

    NFKC expands compatibility characters — U+FB03 becomes three characters —
    so a request that fits the limit as written can exceed it as sent. The
    aggregate is therefore re-checked after normalization, before egress.
    """

    def test_expansion_past_the_limit_is_rejected(self, backend):
        # Each ﬃ normalizes to three characters, so this is under the limit as
        # written and comfortably over it once normalized.
        text = "ﬃ" * (MAX_EMBEDDING_CHARS // 2)
        response = post([{"id": "a", "text": text}])
        assert response.status_code == 422
        assert str(MAX_EMBEDDING_CHARS) in response.json()["detail"]

    def test_nothing_is_embedded_when_the_normalized_form_is_too_large(self, backend):
        text = "ﬃ" * (MAX_EMBEDDING_CHARS // 2)
        assert post([{"id": "a", "text": text}]).status_code == 422
        assert backend.calls == []

    def test_the_limit_is_measured_across_all_inputs(self, backend):
        share = MAX_EMBEDDING_CHARS // 6
        inputs = [{"id": f"i{n}", "text": "ﬃ" * share} for n in range(3)]
        assert post(inputs).status_code == 422
        assert backend.calls == []

    def test_expansion_within_the_limit_still_succeeds(self, backend):
        text = "ﬃ" * (MAX_EMBEDDING_CHARS // 6)
        response = post([{"id": "a", "text": text}])
        assert response.status_code == 200
        assert response.json()["usage"]["total_characters"] == len(text) * 3

    def test_a_request_that_does_not_expand_is_unaffected(self, backend):
        text = "x" * (MAX_EMBEDDING_CHARS - 1)
        assert post([{"id": "a", "text": text}]).status_code == 200


class TestInputTypeIsHonoured:
    """``input_type`` selects the encoder, rather than being recorded and ignored."""

    def test_a_query_goes_through_the_query_encoder_when_one_exists(
        self, backend, monkeypatch
    ):
        seen = {}

        def embed_queries(texts):
            seen["queries"] = list(texts)
            return [deterministic_vector(text) for text in texts]

        monkeypatch.setattr(backend, "embed_queries", embed_queries, raising=False)
        assert (
            post([{"id": "a", "text": "alpha"}], input_type="query").status_code == 200
        )
        assert seen["queries"] == ["alpha"]
        assert backend.calls == []

    def test_a_document_never_goes_through_the_query_encoder(
        self, backend, monkeypatch
    ):
        def explode(texts):
            raise AssertionError("documents must not use the query encoder")

        monkeypatch.setattr(backend, "embed_queries", explode, raising=False)
        response = post([{"id": "a", "text": "alpha"}], input_type="document")
        assert response.status_code == 200
        assert backend.embedded_texts == ["alpha"]

    def test_the_query_task_prefix_reaches_the_backend(self, backend, monkeypatch):
        space = install_fake_space(monkeypatch, backend)
        monkeypatch.setattr(
            space,
            "spec",
            replace(space.spec, query_prefix="Q: ", document_prefix="D: "),
        )
        assert (
            post([{"id": "a", "text": "alpha"}], input_type="query").status_code == 200
        )
        assert backend.embedded_texts == ["Q: alpha"]

    def test_the_document_task_prefix_reaches_the_backend(self, backend, monkeypatch):
        space = install_fake_space(monkeypatch, backend)
        monkeypatch.setattr(
            space,
            "spec",
            replace(space.spec, query_prefix="Q: ", document_prefix="D: "),
        )
        response = post([{"id": "a", "text": "alpha"}], input_type="document")
        assert response.status_code == 200
        assert backend.embedded_texts == ["D: alpha"]

    def test_the_two_input_types_produce_different_vectors(self, backend, monkeypatch):
        space = install_fake_space(monkeypatch, backend)
        monkeypatch.setattr(
            space,
            "spec",
            replace(space.spec, query_prefix="Q: ", document_prefix="D: "),
        )
        as_query = post([{"id": "a", "text": "alpha"}], input_type="query").json()
        as_document = post([{"id": "a", "text": "alpha"}], input_type="document").json()
        assert as_query["items"][0]["embedding"] != as_document["items"][0]["embedding"]

    def test_the_content_hash_identifies_the_text_not_the_encoding(
        self, backend, monkeypatch
    ):
        """The hash is the caller's cache key, so the task prefix stays out of it."""
        space = install_fake_space(monkeypatch, backend)
        monkeypatch.setattr(
            space,
            "spec",
            replace(space.spec, query_prefix="Q: ", document_prefix="D: "),
        )
        as_query = post([{"id": "a", "text": "alpha"}], input_type="query").json()
        as_document = post([{"id": "a", "text": "alpha"}], input_type="document").json()
        assert as_query["items"][0]["content_hash"] == content_hash("alpha")
        assert (
            as_query["items"][0]["content_hash"]
            == as_document["items"][0]["content_hash"]
        )

    def test_an_unprefixed_space_is_unchanged_by_the_input_type(self, backend):
        as_query = post([{"id": "a", "text": "alpha"}], input_type="query").json()
        as_document = post([{"id": "a", "text": "alpha"}], input_type="document").json()
        assert as_query["items"] == as_document["items"]


def test_unknown_space_is_rejected(backend):
    assert post([{"id": "a", "text": "x"}], space="chat-v2").status_code == 400


def test_backend_failure_returns_503_rather_than_a_substitute(backend, monkeypatch):
    install_fake_space(
        monkeypatch, FakeEmbeddingClient(error=RuntimeError("gateway down"))
    )
    response = post([{"id": "a", "text": "x"}])
    assert response.status_code == 503
    assert "gateway down" not in response.text


def test_wrong_dimensionality_is_a_backend_failure(backend, monkeypatch):
    """A model swap behind the gateway must surface, not silently re-dimension."""
    install_fake_space(monkeypatch, FakeEmbeddingClient(dimensions=4), dimensions=8)
    assert post([{"id": "a", "text": "x"}]).status_code == 503


def test_input_text_never_reaches_the_logs(backend, caplog):
    secret = "correcthorsebatterystaple"
    with caplog.at_level(logging.DEBUG):
        assert post([{"id": "a", "text": secret}]).status_code == 200
    assert secret not in caplog.text


def test_input_text_never_reaches_the_logs_on_failure(backend, caplog, monkeypatch):
    install_fake_space(monkeypatch, FakeEmbeddingClient(error=RuntimeError("boom")))
    secret = "correcthorsebatterystaple"
    with caplog.at_level(logging.DEBUG):
        assert post([{"id": "a", "text": secret}]).status_code == 503
    assert secret not in caplog.text


def test_a_provider_that_echoes_the_input_still_leaks_nothing(
    backend, caplog, monkeypatch
):
    """Gateways routinely quote the rejected input back in the error message.

    SpaceBackendError wraps that message verbatim, so logging the exception —
    or its traceback — would write the caller's text into the log despite this
    endpoint's no-raw-text guarantee.
    """
    secret = "correcthorsebatterystaple"
    install_fake_space(
        monkeypatch,
        FakeEmbeddingClient(error=RuntimeError(f"input rejected: '{secret}'")),
    )
    with caplog.at_level(logging.DEBUG):
        response = post([{"id": "a", "text": secret}])
    assert response.status_code == 503
    assert secret not in caplog.text
    assert secret not in response.text
    # The failure is still diagnosable: the class chain identifies the cause.
    assert "SpaceBackendError" in caplog.text
    assert "RuntimeError" in caplog.text


class TestAuthorizeBeforeEgress:
    """Nothing is embedded until the request is fully authorized.

    Text sent to the inference gateway has left the trust boundary whether or
    not the caller ever sees a response, so every rejection has to happen before
    the backend is touched.
    """

    def test_a_token_without_the_embed_scope_embeds_nothing(self, backend):
        response = post(
            [{"id": "a", "text": "secret"}], token=strict_token(scopes=["rag:rerank"])
        )
        assert response.status_code == 403
        assert backend.calls == []

    def test_the_system_tenant_embeds_nothing(self, backend):
        response = post(
            [{"id": "a", "text": "secret"}], token=strict_token(tenant="__SYSTEM__")
        )
        assert response.status_code == 403
        assert backend.calls == []

    def test_an_unauthenticated_request_embeds_nothing(self, backend):
        response = client.post(
            "/v1/embeddings",
            json={
                "space": SPACE,
                "input_type": "query",
                "inputs": [{"id": "a", "text": "secret"}],
            },
        )
        assert response.status_code == 401
        assert backend.calls == []

    def test_a_session_token_embeds_nothing(self, backend):
        response = post(
            [{"id": "a", "text": "secret"}], token=legacy_token(secret=APP_SECRET)
        )
        assert response.status_code == 401
        assert backend.calls == []

    def test_an_over_limit_request_embeds_nothing(self, backend):
        inputs = [{"id": f"i{n}", "text": "x"} for n in range(MAX_EMBEDDING_INPUTS + 1)]
        assert post(inputs).status_code == 422
        assert backend.calls == []

    def test_an_unknown_space_embeds_nothing(self, backend):
        assert post([{"id": "a", "text": "secret"}], space="chat-v2").status_code == 400
        assert backend.calls == []

    def test_a_rate_limited_request_embeds_nothing(self, backend, monkeypatch):
        monkeypatch.setenv("RAG_RATE_LIMIT_ENABLED", "true")
        monkeypatch.setenv("RAG_RATE_LIMIT_EMBED_SUBJECT", "1")
        ratelimit.reset()
        assert post([{"id": "a", "text": "first"}]).status_code == 200
        assert len(backend.calls) == 1
        assert post([{"id": "a", "text": "second"}]).status_code == 429
        assert len(backend.calls) == 1

    def test_the_endpoint_reads_no_document_store_at_all(self, backend, monkeypatch):
        """/v1/embeddings only ever embeds text the caller itself supplied."""
        from app.config import vector_store

        def explode(*args, **kwargs):
            raise AssertionError("the embeddings path must not touch the store")

        for name in (
            "get_vectors_by_ids",
            "probe_candidate_ids",
            "get_documents_by_ids",
        ):
            monkeypatch.setattr(vector_store, name, explode, raising=False)
        assert post([{"id": "a", "text": "hello"}]).status_code == 200
