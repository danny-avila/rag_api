# app/models.py
import hashlib
from enum import Enum
from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Optional

from app.constants import (
    MAX_EMBEDDING_CHARS,
    MAX_EMBEDDING_INPUTS,
    MAX_QUERY_CHARS,
    MAX_RERANK_CANDIDATES,
    MAX_RERANK_TOP_N,
)


class DocumentResponse(BaseModel):
    page_content: str
    metadata: dict


class DocumentModel(BaseModel):
    page_content: str
    metadata: Optional[dict] = {}

    def generate_digest(self):
        hash_obj = hashlib.md5(self.page_content.encode())
        return hash_obj.hexdigest()


class StoreDocument(BaseModel):
    filepath: str
    filename: str
    file_content_type: str
    file_id: str


class QueryRequestBody(BaseModel):
    query: str
    file_id: str
    k: int = 4
    entity_id: Optional[str] = None


class CleanupMethod(str, Enum):
    incremental = "incremental"
    full = "full"


class QueryMultipleBody(BaseModel):
    query: str
    file_ids: List[str]
    k: int = 4
    entity_id: Optional[str] = None


def _reject_duplicates(values: List[str], label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")


class EmbeddingInput(BaseModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class EmbeddingsRequest(BaseModel):
    space: str = Field(min_length=1)
    input_type: Literal["query", "document"]
    inputs: List[EmbeddingInput] = Field(min_length=1, max_length=MAX_EMBEDDING_INPUTS)

    @field_validator("inputs")
    @classmethod
    def _validate_inputs(cls, inputs: List[EmbeddingInput]) -> List[EmbeddingInput]:
        """Reject the request on what the caller sent.

        This is the cheap arm of the size limit. It cannot be the only one:
        NFKC normalization runs in the route and expands compatibility
        characters, so the route re-checks the aggregate length of the
        normalized text before any of it reaches the backend.
        """
        _reject_duplicates([item.id for item in inputs], "input ids")
        total_characters = sum(len(item.text) for item in inputs)
        if total_characters > MAX_EMBEDDING_CHARS:
            raise ValueError(
                f"aggregate input length {total_characters} exceeds "
                f"{MAX_EMBEDDING_CHARS} characters"
            )
        return inputs


class EmbeddingItem(BaseModel):
    id: str
    content_hash: str
    embedding: List[float]


class EmbeddingsUsage(BaseModel):
    input_count: int
    total_characters: int


class EmbeddingsResponse(BaseModel):
    space: str
    model: str
    dimensions: int
    normalized: bool
    items: List[EmbeddingItem]
    usage: EmbeddingsUsage


class RerankCandidate(BaseModel):
    id: str = Field(min_length=1)
    text: Optional[str] = None
    base_score: Optional[float] = None


class RerankRequest(BaseModel):
    profile: str = Field(min_length=1)
    # Bounded here rather than in the route: the query is embedded on its own, so
    # the aggregate candidate budget below never covers it, and an unbounded one
    # reaches the provider and comes back as a 503 instead of a 422.
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    candidates: List[RerankCandidate] = Field(
        min_length=1, max_length=MAX_RERANK_CANDIDATES
    )
    top_n: Optional[int] = Field(default=None, ge=1, le=MAX_RERANK_TOP_N)

    @field_validator("candidates")
    @classmethod
    def _validate_candidates(
        cls, candidates: List[RerankCandidate]
    ) -> List[RerankCandidate]:
        _reject_duplicates([candidate.id for candidate in candidates], "candidate ids")
        total_characters = sum(len(candidate.text or "") for candidate in candidates)
        if total_characters > MAX_EMBEDDING_CHARS:
            raise ValueError(
                f"aggregate candidate length {total_characters} exceeds "
                f"{MAX_EMBEDDING_CHARS} characters"
            )
        return candidates

    def resolved_top_n(self) -> int:
        if self.top_n is None:
            return min(len(self.candidates), MAX_RERANK_TOP_N)
        return min(self.top_n, len(self.candidates))


class RerankResult(BaseModel):
    id: str
    index: int
    score: float


class RerankResponse(BaseModel):
    profile: str
    model: str
    results: List[RerankResult]
