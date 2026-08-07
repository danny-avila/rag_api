"""Per-tenant and per-subject request budgets for the service endpoints.

Embedding and rerank hold separate budgets: a rerank storm must not starve the
interactive query-embedding path, and the backfill (which runs under its own
service subject) must not consume live traffic's allowance.

The counters are process-local. A multi-pod deployment therefore enforces
``limit x pods``; a shared store is the follow-up hardening, and the budgets are
sized as a safety valve rather than a billing control.
"""

import os
import time
from dataclasses import dataclass
from threading import Lock
from typing import Dict, Optional, Tuple

BUDGET_EMBED = "embed"
BUDGET_RERANK = "rerank"

_MAX_TRACKED_KEYS = 20_000


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _flag_env(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("true", "1", "yes", "on", "y")


@dataclass(frozen=True)
class Budget:
    tenant_limit: int
    subject_limit: int
    window_seconds: int


@dataclass(frozen=True)
class RateLimitSettings:
    enabled: bool
    window_seconds: int
    budgets: Dict[str, Budget]

    @classmethod
    def from_env(cls) -> "RateLimitSettings":
        window = _int_env("RAG_RATE_LIMIT_WINDOW_SECONDS", 60)
        return cls(
            enabled=_flag_env("RAG_RATE_LIMIT_ENABLED", "true"),
            window_seconds=window,
            budgets={
                BUDGET_EMBED: Budget(
                    tenant_limit=_int_env("RAG_RATE_LIMIT_EMBED_TENANT", 600),
                    subject_limit=_int_env("RAG_RATE_LIMIT_EMBED_SUBJECT", 120),
                    window_seconds=window,
                ),
                BUDGET_RERANK: Budget(
                    tenant_limit=_int_env("RAG_RATE_LIMIT_RERANK_TENANT", 900),
                    subject_limit=_int_env("RAG_RATE_LIMIT_RERANK_SUBJECT", 180),
                    window_seconds=window,
                ),
            },
        )


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    scope: Optional[str] = None
    retry_after: int = 0


class FixedWindowLimiter:
    """Fixed-window counters keyed by (budget, scope, identity)."""

    def __init__(self):
        self._counters: Dict[Tuple[str, str, str], Tuple[float, int]] = {}
        self._lock = Lock()

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()

    def _hit(self, key, limit: int, window: int, now: float) -> Optional[int]:
        window_start = now - (now % window)
        started, count = self._counters.get(key, (window_start, 0))
        if started < window_start:
            started, count = window_start, 0
        if count >= limit:
            return max(1, int(started + window - now))
        self._counters[key] = (started, count + 1)
        return None

    def _prune(self, now: float, window: int) -> None:
        if len(self._counters) <= _MAX_TRACKED_KEYS:
            return
        cutoff = now - (now % window)
        for key in [
            key for key, (started, _) in self._counters.items() if started < cutoff
        ]:
            del self._counters[key]

    def check(
        self,
        budget_name: str,
        budget: Budget,
        tenant: str,
        subject: str,
        now: Optional[float] = None,
    ) -> RateLimitDecision:
        now = time.time() if now is None else now
        with self._lock:
            self._prune(now, budget.window_seconds)
            retry_after = self._hit(
                (budget_name, "tenant", tenant),
                budget.tenant_limit,
                budget.window_seconds,
                now,
            )
            if retry_after is not None:
                return RateLimitDecision(False, "tenant", retry_after)
            retry_after = self._hit(
                (budget_name, "subject", subject),
                budget.subject_limit,
                budget.window_seconds,
                now,
            )
            if retry_after is not None:
                return RateLimitDecision(False, "subject", retry_after)
        return RateLimitDecision(True)


_limiter = FixedWindowLimiter()
_settings: Optional[RateLimitSettings] = None


def get_settings() -> RateLimitSettings:
    global _settings
    if _settings is None:
        _settings = RateLimitSettings.from_env()
    return _settings


def reset() -> None:
    """Drop cached settings and counters — used between tests."""
    global _settings
    _settings = None
    _limiter.reset()


def check(budget_name: str, tenant: str, subject: str) -> RateLimitDecision:
    settings = get_settings()
    if not settings.enabled:
        return RateLimitDecision(True)
    budget = settings.budgets[budget_name]
    return _limiter.check(budget_name, budget, tenant, subject)
