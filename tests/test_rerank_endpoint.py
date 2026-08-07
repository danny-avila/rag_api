"""POST /v1/rerank — the fast-v1 embed-blend contract.

Candidate vectors come from storage where they exist; only vectorless
candidates are embedded. Scores blend cosine similarity with the caller's
base_score, so ranking is never pure embedding order. No candidate text
appears in the response or in the logs.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import pytest
from fastapi.testclient import TestClient

from app import auth
from app.config import vector_store
from app.constants import (
    MAX_EMBEDDING_CHARS,
    MAX_QUERY_CHARS,
    MAX_RERANK_CANDIDATES,
    MAX_RERANK_TOP_N,
)
from app.services import embedding as embedding_service
from app.services import ratelimit
from main import app
from tests.fakes import FakeEmbeddingClient
from tests.tokens import APP_SECRET, RAG_SECRET, bearer, strict_token

client = TestClient(app)

QUERY = "how do I rotate the signing key"

# Two-dimensional vectors make the cosine order exact and readable.
VECTORS = {
    QUERY: [1.0, 0.0],
    "alpha": [1.0, 0.0],
    "beta": [0.9, 0.4358898943540674],
    "gamma": [0.8, 0.6],
    "delta": [0.0, 1.0],
}


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setenv("RAG_SEARCH_API_ENABLED", "true")
    monkeypatch.setenv("RAG_JWT_SECRET", RAG_SECRET)
    monkeypatch.setenv("JWT_SECRET", APP_SECRET)
    monkeypatch.setenv("RAG_RATE_LIMIT_ENABLED", "false")
    auth.reset_settings()
    if getattr(app.state, "thread_pool", None) is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)

    fake = FakeEmbeddingClient(dimensions=2, vectors=VECTORS)
    monkeypatch.setattr(
        "app.services.embedding.get_cached_query_embedding", fake.embed_query
    )
    monkeypatch.setattr("app.services.embedding.embed_texts", fake.embed_documents)
    monkeypatch.setattr(
        vector_store,
        "get_vectors_by_ids",
        lambda ids, owners, tenants=(None,), executor=None: {},
        raising=False,
    )
    monkeypatch.setattr(
        vector_store,
        "probe_candidate_ids",
        lambda ids, owners, tenants=(None,), executor=None: (set(), set()),
        raising=False,
    )
    return fake


class FakeStore:
    """Store stub that records how it was scoped and answers both lookups.

    ``rows`` maps a candidate id to ``(owner, tenant, vector)`` so the probe and
    the vector lookup answer from one source of truth, exactly as the database
    does.
    """

    def __init__(self):
        self.rows: Dict[str, tuple] = {}
        self.probe_calls: List[Dict[str, object]] = []
        self.vector_calls: List[Dict[str, object]] = []

    def add(self, candidate_id, vector, owner="user-1", tenant="__BASE__"):
        self.rows[candidate_id] = (owner, tenant, vector)

    def probe_candidate_ids(self, ids, owners, tenants=(None,), executor=None):
        self.probe_calls.append(
            {"ids": list(ids), "owners": list(owners), "tenants": list(tenants)}
        )
        existing, authorized = set(), set()
        for candidate_id in ids:
            if candidate_id not in self.rows:
                continue
            owner, tenant, _ = self.rows[candidate_id]
            existing.add(candidate_id)
            if owner in owners and tenant in tenants:
                authorized.add(candidate_id)
        return existing, authorized

    def get_vectors_by_ids(self, ids, owners, tenants=(None,), executor=None):
        self.vector_calls.append(
            {"ids": list(ids), "owners": list(owners), "tenants": list(tenants)}
        )
        found = {}
        for candidate_id in ids:
            if candidate_id not in self.rows:
                continue
            owner, tenant, vector = self.rows[candidate_id]
            if owner in owners and tenant in tenants:
                found[candidate_id] = vector
        return found


@pytest.fixture
def stored(monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(
        vector_store, "probe_candidate_ids", store.probe_candidate_ids, raising=False
    )
    monkeypatch.setattr(
        vector_store, "get_vectors_by_ids", store.get_vectors_by_ids, raising=False
    )
    return store


def rerank(candidates, profile="fast-v1", top_n=None, token=None, query=QUERY):
    body = {"profile": profile, "query": query, "candidates": candidates}
    if top_n is not None:
        body["top_n"] = top_n
    return client.post("/v1/rerank", json=body, headers=bearer(token or strict_token()))


def blended_candidates():
    return [
        {"id": "c-alpha", "text": "alpha", "base_score": 1.0},
        {"id": "c-beta", "text": "beta", "base_score": 5.0},
        {"id": "c-gamma", "text": "gamma", "base_score": 10.0},
    ]


def test_response_shape_matches_the_contract(backend):
    response = rerank(blended_candidates())
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"profile", "model", "results"}
    assert payload["profile"] == "fast-v1"
    assert payload["model"] == embedding_service.STORE_EMBEDDING_MODEL
    assert all(set(result) == {"id", "index", "score"} for result in payload["results"])


def test_caller_ids_and_request_indices_are_preserved(backend):
    payload = rerank(blended_candidates()).json()
    for result in payload["results"]:
        assert blended_candidates()[result["index"]]["id"] == result["id"]


def test_ranking_is_not_pure_embedding_order(backend):
    """Cosine order here is alpha > beta > gamma; the base scores reverse beta/gamma."""
    payload = rerank(blended_candidates()).json()
    assert [result["id"] for result in payload["results"]] == [
        "c-alpha",
        "c-gamma",
        "c-beta",
    ]


def test_exact_ties_order_by_request_position(backend):
    payload = rerank(blended_candidates()).json()
    scores = {result["id"]: result["score"] for result in payload["results"]}
    assert scores["c-alpha"] == scores["c-gamma"]
    indices = [result["index"] for result in payload["results"]]
    assert indices[0] < indices[1]


def test_repeated_requests_are_identical(backend):
    first = rerank(blended_candidates()).json()
    second = rerank(blended_candidates()).json()
    assert first == second


def test_candidate_text_never_appears_in_the_response(backend):
    response = rerank(
        [
            {"id": "c1", "text": "confidential chunk body", "base_score": 1.0},
            {"id": "c2", "text": "another private passage", "base_score": 2.0},
        ]
    )
    assert response.status_code == 200
    assert "confidential" not in response.text
    assert "private" not in response.text
    assert all(
        set(result) == {"id", "index", "score"} for result in response.json()["results"]
    )


def test_query_and_candidate_text_never_reach_the_logs(backend, caplog):
    secret_query = "supersecretquerystring"
    secret_text = "supersecretcandidatetext"
    with caplog.at_level(logging.DEBUG):
        response = rerank(
            [{"id": "c1", "text": secret_text, "base_score": 1.0}], query=secret_query
        )
    assert response.status_code == 200
    assert secret_query not in caplog.text
    assert secret_text not in caplog.text


def test_top_n_truncates(backend):
    payload = rerank(blended_candidates(), top_n=2).json()
    assert len(payload["results"]) == 2


def test_top_n_defaults_to_the_candidate_count(backend):
    payload = rerank(blended_candidates()).json()
    assert len(payload["results"]) == 3


def test_top_n_above_the_cap_is_rejected(backend):
    response = rerank(blended_candidates(), top_n=MAX_RERANK_TOP_N + 1)
    assert response.status_code == 422


def test_default_top_n_never_exceeds_the_cap(backend):
    candidates = [
        {"id": f"c{n}", "text": f"text {n}", "base_score": float(n)}
        for n in range(MAX_RERANK_CANDIDATES)
    ]
    payload = rerank(candidates).json()
    assert len(payload["results"]) == MAX_RERANK_TOP_N


def test_candidate_limit_is_enforced(backend):
    candidates = [
        {"id": f"c{n}", "text": "t", "base_score": 1.0}
        for n in range(MAX_RERANK_CANDIDATES + 1)
    ]
    assert rerank(candidates).status_code == 422


def test_duplicate_candidate_ids_are_rejected(backend):
    candidates = [
        {"id": "same", "text": "alpha", "base_score": 1.0},
        {"id": "same", "text": "beta", "base_score": 1.0},
    ]
    assert rerank(candidates).status_code == 422


def test_unknown_profile_is_rejected(backend):
    assert rerank(blended_candidates(), profile="slow-v9").status_code == 400


def test_stored_vectors_are_reused_and_only_gaps_are_embedded(backend, stored):
    stored.add("c-alpha", VECTORS["alpha"])
    stored.add("c-beta", VECTORS["beta"])

    response = rerank(blended_candidates())
    assert response.status_code == 200
    # Query embed + the one vectorless candidate, and nothing else.
    assert backend.embedded_texts == [QUERY, "gamma"]
    assert stored.vector_calls[0]["ids"] == ["c-alpha", "c-beta", "c-gamma"]


def test_fully_stored_candidates_cost_no_candidate_inference(backend, stored):
    for candidate_id, text in (
        ("c-alpha", "alpha"),
        ("c-beta", "beta"),
        ("c-gamma", "gamma"),
    ):
        stored.add(candidate_id, VECTORS[text])

    assert rerank(blended_candidates()).status_code == 200
    assert backend.embedded_texts == [QUERY]


def test_stored_lookup_is_scoped_to_the_tokens_owners(backend, stored):
    token = strict_token(subject="user-1", entities=["agent-7", "agent-9"])
    assert rerank(blended_candidates(), token=token).status_code == 200
    assert stored.vector_calls[0]["owners"] == ["agent-7", "agent-9", "user-1"]


def test_stored_lookup_is_scoped_to_the_subject_alone_without_entities(backend, stored):
    assert rerank(blended_candidates()).status_code == 200
    assert stored.vector_calls[0]["owners"] == ["user-1"]


def test_lookup_is_scoped_to_the_tokens_tenant(backend, stored):
    token = strict_token(subject="user-1", tenant="tenant-a")
    assert rerank(blended_candidates(), token=token).status_code == 200
    assert stored.vector_calls[0]["tenants"] == ["tenant-a"]
    assert stored.probe_calls[0]["tenants"] == ["tenant-a"]


def test_base_tenant_also_matches_chunks_written_before_tenants_were_recorded(
    backend, stored
):
    assert rerank(blended_candidates()).status_code == 200
    assert stored.vector_calls[0]["tenants"] == ["__BASE__", None]


def test_vector_lookup_failure_degrades_to_inference(backend, stored, monkeypatch):
    """Losing stored vectors costs inference; it does not lose authorization."""

    def explode(ids, owners, tenants=(None,), executor=None):
        raise RuntimeError("store is down")

    monkeypatch.setattr(vector_store, "get_vectors_by_ids", explode, raising=False)
    response = rerank(blended_candidates())
    assert response.status_code == 200
    assert len(response.json()["results"]) == 3


class TestAuthorizeBeforeEgress:
    """No candidate text may reach the gateway before scope is verified."""

    def test_a_foreign_candidate_is_refused_and_nothing_is_embedded(
        self, backend, stored
    ):
        stored.add("c-alpha", VECTORS["alpha"], owner="user-2")
        response = rerank(blended_candidates())
        assert response.status_code == 403
        assert backend.calls == []

    def test_a_cross_tenant_candidate_is_refused_and_nothing_is_embedded(
        self, backend, stored
    ):
        stored.add("c-alpha", VECTORS["alpha"], owner="user-1", tenant="tenant-b")
        response = rerank(
            blended_candidates(),
            token=strict_token(subject="user-1", tenant="tenant-a"),
        )
        assert response.status_code == 403
        assert backend.calls == []

    def test_authorization_runs_before_the_vector_lookup(self, backend, stored):
        stored.add("c-alpha", VECTORS["alpha"], owner="user-2")
        assert rerank(blended_candidates()).status_code == 403
        assert stored.probe_calls != []
        assert stored.vector_calls == []

    def test_ids_that_match_nothing_in_the_store_are_unaffected(self, backend, stored):
        candidates = [{"id": "web-scrape-1", "text": "gamma", "base_score": 1.0}]
        assert rerank(candidates).status_code == 200
        assert backend.embedded_texts == [QUERY, "gamma"]

    def test_a_digest_shared_with_another_owner_is_still_authorized(
        self, backend, stored
    ):
        """Identical chunk content collides on digest; owning a copy is enough."""
        stored.add("c-alpha", VECTORS["alpha"], owner="user-1")
        assert rerank(blended_candidates()).status_code == 200

    def test_probe_failure_fails_closed_rather_than_embedding(
        self, backend, stored, monkeypatch
    ):
        def explode(ids, owners, tenants=(None,), executor=None):
            raise RuntimeError("store is down")

        monkeypatch.setattr(vector_store, "probe_candidate_ids", explode, raising=False)
        response = rerank(blended_candidates())
        assert response.status_code == 503
        assert backend.calls == []
        assert "store is down" not in response.text

    def test_unauthorized_requests_never_reach_the_backend(self, backend, stored):
        no_scope = strict_token(subject="user-1", scopes=["rag:embed"])
        assert rerank(blended_candidates(), token=no_scope).status_code == 403
        system = strict_token(subject="user-1", tenant="__SYSTEM__")
        assert rerank(blended_candidates(), token=system).status_code == 403
        assert rerank(blended_candidates(), profile="slow-v9").status_code == 400
        assert backend.calls == []
        assert stored.probe_calls == []

    def test_rate_limited_requests_never_reach_the_backend(
        self, backend, stored, monkeypatch
    ):
        monkeypatch.setenv("RAG_RATE_LIMIT_ENABLED", "true")
        monkeypatch.setenv("RAG_RATE_LIMIT_RERANK_SUBJECT", "1")
        ratelimit.reset()
        assert rerank(blended_candidates()).status_code == 200
        calls_after_first = len(backend.calls)
        assert rerank(blended_candidates()).status_code == 429
        assert len(backend.calls) == calls_after_first


def test_embedding_backend_failure_returns_503(backend, monkeypatch):
    def explode(text):
        raise RuntimeError("gateway down")

    monkeypatch.setattr("app.services.embedding.get_cached_query_embedding", explode)
    response = rerank(blended_candidates())
    assert response.status_code == 503
    assert "gateway down" not in response.text


class TestFailuresAreLoggedWithoutProviderText:
    """An exception message is provider-controlled and may quote the input.

    The endpoint promises that no query or candidate text is logged. That has to
    hold on the error paths too, which means logging the exception's class chain
    rather than its message or a traceback.
    """

    SECRET_QUERY = "supersecretquerystring"
    SECRET_TEXT = "supersecretcandidatetext"

    def _candidates(self):
        return [{"id": "c1", "text": self.SECRET_TEXT, "base_score": 1.0}]

    def _assert_clean(self, caplog, response):
        assert self.SECRET_QUERY not in caplog.text
        assert self.SECRET_TEXT not in caplog.text
        assert self.SECRET_QUERY not in response.text
        assert self.SECRET_TEXT not in response.text

    def test_a_query_embedding_failure_that_echoes_the_query(
        self, backend, caplog, monkeypatch
    ):
        def explode(text):
            raise RuntimeError(f"tokenizer rejected: '{text}'")

        monkeypatch.setattr(
            "app.services.embedding.get_cached_query_embedding", explode
        )
        with caplog.at_level(logging.DEBUG):
            response = rerank(self._candidates(), query=self.SECRET_QUERY)
        assert response.status_code == 503
        self._assert_clean(caplog, response)
        assert "RuntimeError" in caplog.text

    def test_a_candidate_embedding_failure_that_echoes_the_candidate(
        self, backend, caplog, monkeypatch
    ):
        def explode(texts):
            raise RuntimeError(f"batch rejected: {texts}")

        monkeypatch.setattr("app.services.embedding.embed_texts", explode)
        with caplog.at_level(logging.DEBUG):
            response = rerank(self._candidates(), query=self.SECRET_QUERY)
        assert response.status_code == 503
        self._assert_clean(caplog, response)

    def test_no_traceback_is_written(self, backend, caplog, monkeypatch):
        """A traceback carries the frame locals' repr into the log with it."""

        def explode(text):
            raise RuntimeError("gateway down")

        monkeypatch.setattr(
            "app.services.embedding.get_cached_query_embedding", explode
        )
        with caplog.at_level(logging.DEBUG):
            assert rerank(self._candidates()).status_code == 503
        assert "Traceback (most recent call last)" not in caplog.text
        assert "gateway down" not in caplog.text

    def test_a_store_failure_does_not_log_the_candidate_ids(
        self, backend, stored, caplog, monkeypatch
    ):
        """Driver errors embed the statement parameters, which are caller strings."""

        def explode(ids, owners, tenants=(None,), executor=None):
            raise RuntimeError(f"could not execute: {list(ids)}")

        monkeypatch.setattr(vector_store, "probe_candidate_ids", explode, raising=False)
        with caplog.at_level(logging.DEBUG):
            response = rerank([{"id": "leaky-id-42", "text": "t", "base_score": 1.0}])
        assert response.status_code == 503
        assert "leaky-id-42" not in caplog.text
        assert "could not execute" not in caplog.text

    def test_the_error_category_chain_survives_wrapping(self):
        from app.routes.search_routes import error_category

        cause = ValueError("provider echoed the input")
        try:
            try:
                raise cause
            except ValueError as exc:
                raise RuntimeError("wrapper") from exc
        except RuntimeError as exc:
            category = error_category(exc)
        assert category == "RuntimeError <- ValueError"
        assert "provider echoed" not in category


def test_candidates_without_text_or_stored_vector_still_rank(backend, stored):
    candidates = [
        {"id": "c-alpha", "text": "alpha", "base_score": 3.0},
        {"id": "c-vectorless", "base_score": 100.0},
        {"id": "c-gamma", "text": "gamma", "base_score": 1.0},
    ]
    payload = rerank(candidates).json()
    returned = [result["id"] for result in payload["results"]]
    assert returned == ["c-alpha", "c-vectorless", "c-gamma"]
    assert backend.embedded_texts == [QUERY, "alpha", "gamma"]


def test_base_scores_are_optional(backend):
    candidates = [
        {"id": "c-gamma", "text": "gamma"},
        {"id": "c-alpha", "text": "alpha"},
    ]
    payload = rerank(candidates).json()
    assert [result["id"] for result in payload["results"]] == ["c-alpha", "c-gamma"]


def test_a_store_that_cannot_be_probed_refuses_rerank(backend, monkeypatch):
    """There is no unauthenticated fallback path for candidate authorization."""
    monkeypatch.delattr(vector_store, "probe_candidate_ids", raising=False)
    response = rerank(blended_candidates())
    assert response.status_code == 503
    assert backend.calls == []


class TestTheQueryIsBounded:
    """The query is embedded on its own, so the candidate budget never covers it.

    Without a limit of its own an authenticated caller hands the gateway an
    arbitrarily large query and gets the provider's refusal back as a 503 —
    inference paid for, memory spent, and a server-fault status for a request
    the service could have rejected.
    """

    def test_a_query_at_the_limit_is_accepted(self, backend, stored):
        response = rerank(blended_candidates(), query="q" * MAX_QUERY_CHARS)
        assert response.status_code == 200

    def test_an_oversized_query_is_rejected(self, backend, stored):
        response = rerank(blended_candidates(), query="q" * (MAX_QUERY_CHARS + 1))
        assert response.status_code == 422

    def test_an_oversized_query_never_reaches_the_backend(self, backend, stored):
        assert (
            rerank(blended_candidates(), query="q" * (MAX_QUERY_CHARS + 1)).status_code
            == 422
        )
        assert backend.calls == []

    def test_the_bound_holds_even_when_the_candidates_are_tiny(self, backend, stored):
        """The candidate aggregate is the check the query used to hide behind."""
        candidates = [{"id": "c-alpha", "text": "a"}]
        response = rerank(candidates, query="q" * (MAX_EMBEDDING_CHARS - 1))
        assert response.status_code == 422
        assert backend.calls == []

    def test_an_empty_query_is_still_rejected(self, backend, stored):
        assert rerank(blended_candidates(), query="").status_code == 422
