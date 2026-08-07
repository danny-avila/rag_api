"""Per-tenant and per-subject budgets, separate for embedding and rerank."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app import auth
from app.config import vector_store
from app.services import ratelimit
from app.services import space as space_module
from main import app
from tests.fakes import FakeEmbeddingClient, install_fake_space
from tests.tokens import APP_SECRET, RAG_SECRET, bearer, strict_token

client = TestClient(app)

SPACE = space_module.CHAT_SPACE_SPEC.name


@pytest.fixture
def limited(monkeypatch):
    monkeypatch.setenv("RAG_SEARCH_API_ENABLED", "true")
    monkeypatch.setenv("RAG_JWT_SECRET", RAG_SECRET)
    monkeypatch.setenv("JWT_SECRET", APP_SECRET)
    monkeypatch.setenv("RAG_RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("RAG_RATE_LIMIT_EMBED_SUBJECT", "2")
    monkeypatch.setenv("RAG_RATE_LIMIT_EMBED_TENANT", "3")
    monkeypatch.setenv("RAG_RATE_LIMIT_RERANK_SUBJECT", "5")
    monkeypatch.setenv("RAG_RATE_LIMIT_RERANK_TENANT", "50")
    auth.reset_settings()
    ratelimit.reset()
    if getattr(app.state, "thread_pool", None) is None:
        app.state.thread_pool = ThreadPoolExecutor(max_workers=2)

    fake = FakeEmbeddingClient(dimensions=2)
    install_fake_space(monkeypatch, fake)
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


def embed(subject="user-1", tenant="tenant-a"):
    return client.post(
        "/v1/embeddings",
        json={
            "space": SPACE,
            "input_type": "query",
            "inputs": [{"id": "a", "text": "x"}],
        },
        headers=bearer(strict_token(subject=subject, tenant=tenant)),
    )


def do_rerank(subject="user-1", tenant="tenant-a"):
    return client.post(
        "/v1/rerank",
        json={
            "profile": "fast-v1",
            "query": "q",
            "candidates": [{"id": "c1", "text": "t", "base_score": 1.0}],
        },
        headers=bearer(strict_token(subject=subject, tenant=tenant)),
    )


def test_subject_budget_is_enforced(limited):
    assert embed().status_code == 200
    assert embed().status_code == 200
    response = embed()
    assert response.status_code == 429
    assert response.headers["Retry-After"]
    assert "subject" in response.json()["detail"]


def test_tenant_budget_is_enforced_across_subjects(limited):
    assert embed(subject="user-1").status_code == 200
    assert embed(subject="user-2").status_code == 200
    assert embed(subject="user-3").status_code == 200
    response = embed(subject="user-4")
    assert response.status_code == 429
    assert "tenant" in response.json()["detail"]


def test_tenants_have_independent_budgets(limited):
    """Three distinct subjects, because only served requests spend the budget.

    The tenant limit is 3 and the subject limit is 2, so one subject cannot
    exhaust the tenant on its own — its third request is rejected by the subject
    arm and never reaches the tenant counter.
    """
    for subject in ("user-1", "user-2", "user-3"):
        assert embed(subject=subject, tenant="tenant-a").status_code == 200
    assert embed(subject="user-9", tenant="tenant-a").status_code == 429
    assert embed(subject="user-9", tenant="tenant-b").status_code == 200


def test_embedding_and_rerank_budgets_are_separate(limited):
    assert embed().status_code == 200
    assert embed().status_code == 200
    assert embed().status_code == 429
    # The rerank budget is untouched by the exhausted embedding budget.
    assert do_rerank().status_code == 200


def test_disabled_limiter_lets_everything_through(limited, monkeypatch):
    monkeypatch.setenv("RAG_RATE_LIMIT_ENABLED", "false")
    ratelimit.reset()
    for _ in range(10):
        assert embed().status_code == 200


class TestLimiterUnit:
    def _budget(self, tenant_limit=10, subject_limit=2, window=60):
        return ratelimit.Budget(
            tenant_limit=tenant_limit,
            subject_limit=subject_limit,
            window_seconds=window,
        )

    def test_counters_reset_in_the_next_window(self):
        limiter = ratelimit.FixedWindowLimiter()
        budget = self._budget()
        now = 1_000_000.0
        assert limiter.check("embed", budget, "t", "s", now).allowed
        assert limiter.check("embed", budget, "t", "s", now).allowed
        assert not limiter.check("embed", budget, "t", "s", now).allowed
        assert limiter.check("embed", budget, "t", "s", now + 60).allowed

    def test_retry_after_points_at_the_next_window(self):
        limiter = ratelimit.FixedWindowLimiter()
        budget = self._budget(subject_limit=1)
        now = 1_000_000.0
        limiter.check("embed", budget, "t", "s", now)
        decision = limiter.check("embed", budget, "t", "s", now + 10)
        assert not decision.allowed
        assert 1 <= decision.retry_after <= 60

    def test_the_tenant_arm_is_checked_before_the_subject_arm(self):
        limiter = ratelimit.FixedWindowLimiter()
        budget = self._budget(tenant_limit=1, subject_limit=5)
        now = 1_000_000.0
        assert limiter.check("embed", budget, "t", "s1", now).allowed
        decision = limiter.check("embed", budget, "t", "s2", now)
        assert decision.scope == "tenant"

    def test_a_subject_rejection_does_not_spend_the_tenant_budget(self):
        """One subject must not be able to deny service to its whole tenant.

        The tenant counter used to increment before the subject arm ran, so a
        single token could burn every tenant slot on requests that were rejected
        anyway and lock out every other subject in that tenant.
        """
        limiter = ratelimit.FixedWindowLimiter()
        budget = self._budget(tenant_limit=4, subject_limit=1)
        now = 1_000_000.0

        assert limiter.check("embed", budget, "t", "noisy", now).allowed
        for _ in range(20):
            decision = limiter.check("embed", budget, "t", "noisy", now)
            assert not decision.allowed
            assert decision.scope == "subject"

        # Three tenant slots are left: the noisy subject spent exactly one.
        for quiet in ("s1", "s2", "s3"):
            assert limiter.check("embed", budget, "t", quiet, now).allowed
        assert limiter.check("embed", budget, "t", "s4", now).scope == "tenant"

    def test_a_tenant_rejection_does_not_spend_the_subject_budget(self):
        limiter = ratelimit.FixedWindowLimiter()
        budget = self._budget(tenant_limit=1, subject_limit=2)
        now = 1_000_000.0

        assert limiter.check("embed", budget, "t", "s1", now).allowed
        for _ in range(5):
            assert limiter.check("embed", budget, "t", "s2", now).scope == "tenant"

        # s2 was never served, so its own budget is untouched in the next window.
        assert limiter.check("embed", budget, "t", "s2", now + 60).allowed
        assert limiter.check("embed", budget, "t", "s2", now + 60).scope == "tenant"


def test_a_throttled_subject_does_not_lock_out_its_tenant(limited):
    """End to end: subject limit 2, tenant limit 3, one noisy caller."""
    assert embed(subject="noisy").status_code == 200
    assert embed(subject="noisy").status_code == 200
    for _ in range(10):
        response = embed(subject="noisy")
        assert response.status_code == 429
        assert "subject" in response.json()["detail"]

    assert embed(subject="quiet").status_code == 200
