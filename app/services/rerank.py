"""``fast-v1`` reranking — embed-blend v0.

Scoring is reciprocal-rank fusion of two arms: cosine similarity between the
query vector and each candidate vector, and the caller's own ``base_score``
(the retrieval score that produced the candidate). Pure bi-encoder order is
deliberately *not* used — it regresses identifier and exact-match queries, and
the blend recovers that loss.

Candidate vectors come from storage wherever they exist; inference is spent
only on candidates that arrive vectorless.
"""

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


@dataclass(frozen=True)
class BlendConfig:
    k: int
    similarity_weight: float
    base_weight: float

    @classmethod
    def from_env(cls) -> "BlendConfig":
        return cls(
            k=_int_env("RAG_RERANK_RRF_K", 60),
            similarity_weight=_float_env("RAG_RERANK_SIMILARITY_WEIGHT", 1.0),
            base_weight=_float_env("RAG_RERANK_BASE_WEIGHT", 1.0),
        )


@dataclass(frozen=True)
class RankedCandidate:
    index: int
    score: float


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    """Cosine similarity, or ``None`` when either side is degenerate."""
    if not left or not right or len(left) != len(right):
        return None
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


def rank_positions(values: Sequence[Optional[float]]) -> List[int]:
    """1-based ranks, highest value first.

    ``None`` entries rank after every scored entry, and ties break on the
    candidate's position in the request — so identical inputs always produce
    identical ranks.
    """
    order = sorted(
        range(len(values)),
        key=lambda index: (
            values[index] is None,
            -(values[index] if values[index] is not None else 0.0),
            index,
        ),
    )
    ranks = [0] * len(values)
    for position, index in enumerate(order, start=1):
        ranks[index] = position
    return ranks


def blend_scores(
    similarities: Sequence[Optional[float]],
    base_scores: Sequence[Optional[float]],
    config: BlendConfig,
) -> List[float]:
    """RRF-blend the similarity arm with the caller's retrieval arm.

    An arm with no usable values at all is dropped rather than contributing a
    constant, so a fully vectorless call degrades to the caller's own order
    instead of shuffling it.
    """
    count = len(similarities)
    use_similarity = any(value is not None for value in similarities)
    use_base = any(value is not None for value in base_scores)

    similarity_ranks = rank_positions(similarities) if use_similarity else None
    base_ranks = rank_positions(base_scores) if use_base else None

    scores = []
    for index in range(count):
        score = 0.0
        if similarity_ranks is not None:
            score += config.similarity_weight / (config.k + similarity_ranks[index])
        if base_ranks is not None:
            score += config.base_weight / (config.k + base_ranks[index])
        scores.append(score)
    return scores


def order_by_score(scores: Sequence[float], top_n: int) -> List[RankedCandidate]:
    """Highest score first, ties broken by request position, truncated to ``top_n``."""
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    return [
        RankedCandidate(index=index, score=scores[index]) for index in order[:top_n]
    ]
