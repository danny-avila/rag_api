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
from app.constants import MAX_RERANK_CANDIDATES, MAX_RERANK_TOP_N
from app.services import embedding as embedding_service
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
        vector_store, "get_vectors_by_ids", lambda ids, owners: {}, raising=False
    )
    return fake


@pytest.fixture
def stored(monkeypatch):
    """Install a store stub and record the (ids, owners) it is queried with."""
    calls: List[Dict[str, List[str]]] = []
    contents: Dict[str, List[float]] = {}

    def lookup(ids, owners):
        calls.append({"ids": list(ids), "owners": list(owners)})
        return {
            candidate_id: contents[candidate_id]
            for candidate_id in ids
            if candidate_id in contents
        }

    monkeypatch.setattr(vector_store, "get_vectors_by_ids", lookup, raising=False)
    return {"calls": calls, "contents": contents}


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
    stored["contents"]["c-alpha"] = VECTORS["alpha"]
    stored["contents"]["c-beta"] = VECTORS["beta"]

    response = rerank(blended_candidates())
    assert response.status_code == 200
    # Query embed + the one vectorless candidate, and nothing else.
    assert backend.embedded_texts == [QUERY, "gamma"]
    assert stored["calls"][0]["ids"] == ["c-alpha", "c-beta", "c-gamma"]


def test_fully_stored_candidates_cost_no_candidate_inference(backend, stored):
    for candidate_id, text in (
        ("c-alpha", "alpha"),
        ("c-beta", "beta"),
        ("c-gamma", "gamma"),
    ):
        stored["contents"][candidate_id] = VECTORS[text]

    assert rerank(blended_candidates()).status_code == 200
    assert backend.embedded_texts == [QUERY]


def test_stored_lookup_is_scoped_to_the_tokens_owners(backend, stored):
    token = strict_token(subject="user-1", entities=["agent-7", "agent-9"])
    assert rerank(blended_candidates(), token=token).status_code == 200
    assert stored["calls"][0]["owners"] == ["agent-7", "agent-9", "user-1"]


def test_stored_lookup_is_scoped_to_the_subject_alone_without_entities(backend, stored):
    assert rerank(blended_candidates()).status_code == 200
    assert stored["calls"][0]["owners"] == ["user-1"]


def test_store_failure_degrades_to_inference_rather_than_failing(backend, monkeypatch):
    def explode(ids, owners):
        raise RuntimeError("store is down")

    monkeypatch.setattr(vector_store, "get_vectors_by_ids", explode, raising=False)
    response = rerank(blended_candidates())
    assert response.status_code == 200
    assert len(response.json()["results"]) == 3


def test_embedding_backend_failure_returns_503(backend, monkeypatch):
    def explode(text):
        raise RuntimeError("gateway down")

    monkeypatch.setattr("app.services.embedding.get_cached_query_embedding", explode)
    response = rerank(blended_candidates())
    assert response.status_code == 503
    assert "gateway down" not in response.text


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
