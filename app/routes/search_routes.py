# app/routes/search_routes.py
"""Strict service endpoints for embedding and reranking.

``POST /v1/embeddings`` serves the substitution-locked ``chat-v1`` space.
``POST /v1/rerank`` serves the ``fast-v1`` profile as embed-blend.

Neither endpoint returns candidate text, and neither logs query or candidate
text — only lengths and non-reversible fingerprints.
"""

import asyncio
import inspect
import traceback
from typing import Dict, List, Optional, Sequence

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.auth import SCOPE_EMBED, SCOPE_RERANK, Principal, require_scope
from app.config import logger, vector_store
from app.constants import (
    MAX_EMBEDDING_CHARS,
    MAX_EMBEDDING_INPUTS,
    RERANK_PROFILE_FAST_V1,
)
from app.models import (
    EmbeddingItem,
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingsUsage,
    RerankRequest,
    RerankResponse,
    RerankResult,
)
from app.scope import ScopeFilter, token_scope
from app.services.vector_store.async_pg_vector import AsyncPgVector
from app.services import embedding as embedding_service
from app.services import ratelimit
from app.services.rerank import (
    BlendConfig,
    blend_scores,
    cosine_similarity,
    order_by_score,
)
from app.services.space import SpaceBackendError, get_space, known_spaces
from app.utils.text import content_hash, fingerprint, normalize_text

router = APIRouter(prefix="/v1")


def _executor(request: Request):
    return getattr(request.app.state, "thread_pool", None)


async def _run(request: Request, func, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor(request), lambda: func(*args))


def _enforce_budget(budget_name: str, principal: Principal) -> None:
    decision = ratelimit.check(budget_name, principal.tenant, principal.subject)
    if decision.allowed:
        return
    logger.warning(
        "Rate limit exceeded | budget=%s scope=%s tenant=%s subject=%s",
        budget_name,
        decision.scope,
        principal.tenant,
        principal.subject,
    )
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=f"{budget_name} rate limit exceeded for this {decision.scope}",
        headers={"Retry-After": str(decision.retry_after)},
    )


@router.post("/embeddings", response_model=EmbeddingsResponse)
async def create_embeddings(
    body: EmbeddingsRequest,
    request: Request,
    principal: Principal = Depends(require_scope(SCOPE_EMBED)),
) -> EmbeddingsResponse:
    space = get_space(body.space)
    if space is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown embedding space '{body.space}' (known: {known_spaces()})",
        )

    _enforce_budget(ratelimit.BUDGET_EMBED, principal)

    texts = [normalize_text(item.text) for item in body.inputs]
    empty = [body.inputs[index].id for index, text in enumerate(texts) if not text]
    if empty:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Inputs normalize to empty text: {empty}",
        )

    try:
        vectors = await _run(request, space.embed_documents, texts)
    except SpaceBackendError as exc:
        # chat-v1 is substitution-locked: a backend failure is a 503, never a
        # quiet switch to another model or dimensionality.
        logger.error(
            "Embedding backend unavailable | space=%s inputs=%d | %s",
            body.space,
            len(texts),
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embedding space '{body.space}' is unavailable",
        )

    items = [
        EmbeddingItem(
            id=body.inputs[index].id,
            content_hash=content_hash(texts[index]),
            embedding=vectors[index],
        )
        for index in range(len(texts))
    ]

    return EmbeddingsResponse(
        space=space.spec.name,
        model=space.spec.model,
        dimensions=space.spec.dimensions,
        normalized=space.spec.normalized,
        items=items,
        usage=EmbeddingsUsage(
            input_count=len(items),
            total_characters=sum(len(text) for text in texts),
        ),
    )


async def _call_store(request: Request, method_name: str, *args):
    """Invoke a store method, handing the async store this request's executor.

    The async store caches ``loop._default_executor`` when none is supplied, and
    that executor belongs to whichever event loop ran first — it is already shut
    down by the next request.
    """
    method = getattr(vector_store, method_name, None)
    if method is None:
        return None
    if isinstance(vector_store, AsyncPgVector):
        result = method(*args, executor=_executor(request))
    else:
        result = method(*args)
    if inspect.isawaitable(result):
        return await result
    return result


async def _authorize_candidates(
    request: Request, ids: Sequence[str], scope: ScopeFilter
) -> None:
    """Refuse candidates that exist in the store outside the caller's scope.

    This runs *before* any text leaves the process. A caller who names another
    owner's chunk and supplies its text must not have that text embedded: the
    unauthorized content would leave the trust boundary even though the caller
    never sees it in a response. Ids that match nothing in the store —
    web-scrape candidates, synthetic ids — are unaffected.

    The probe reads metadata only, never vectors or document text. If it cannot
    run, the request fails closed: without it there is no way to tell a foreign
    candidate from a fresh one.
    """
    if not hasattr(vector_store, "probe_candidate_ids"):
        return
    try:
        probed = await _call_store(
            request,
            "probe_candidate_ids",
            list(ids),
            list(scope.owners),
            scope.tenant_values(),
        )
        existing, authorized = probed
    except Exception as exc:
        logger.error(
            "Candidate authorization probe failed | %s: %s", type(exc).__name__, exc
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Candidate authorization is unavailable",
        )

    foreign = sorted(existing - authorized)
    if foreign:
        logger.warning(
            "Refused out-of-scope rerank candidates | tenant=%s owners=%s count=%d",
            scope.tenant,
            ",".join(scope.owners),
            len(foreign),
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Candidates outside the caller's scope: {foreign}",
        )


async def _stored_vectors(
    request: Request, ids: Sequence[str], scope: ScopeFilter
) -> Dict[str, List[float]]:
    """Candidate vectors already in the store, scoped to the caller's tenant and owners.

    A store failure degrades to "no stored vectors" — the vectorless path still
    produces a correct ranking, at the cost of candidate inference. That is safe
    only because :func:`_authorize_candidates` has already run and fails closed.
    """
    if not hasattr(vector_store, "get_vectors_by_ids") or not scope.owners:
        return {}
    try:
        return (
            await _call_store(
                request,
                "get_vectors_by_ids",
                list(ids),
                list(scope.owners),
                scope.tenant_values(),
            )
            or {}
        )
    except Exception as exc:
        logger.warning(
            "Stored candidate vector lookup failed, falling back to inference | %s: %s",
            type(exc).__name__,
            exc,
        )
        return {}


@router.post("/rerank", response_model=RerankResponse)
async def rerank(
    body: RerankRequest,
    request: Request,
    principal: Principal = Depends(require_scope(SCOPE_RERANK)),
) -> RerankResponse:
    if body.profile != RERANK_PROFILE_FAST_V1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown rerank profile '{body.profile}'",
        )

    _enforce_budget(ratelimit.BUDGET_RERANK, principal)

    # Authorize before egress: every candidate is scope-checked against the
    # store before a single character of candidate text reaches the gateway.
    scope = token_scope(request)
    candidate_ids = [candidate.id for candidate in body.candidates]
    await _authorize_candidates(request, candidate_ids, scope)
    stored = await _stored_vectors(request, candidate_ids, scope)

    pending: List[int] = []
    pending_texts: List[str] = []
    for index, candidate in enumerate(body.candidates):
        if candidate.id in stored or not candidate.text:
            continue
        text = normalize_text(candidate.text)
        if not text:
            continue
        pending.append(index)
        pending_texts.append(text)

    if len(pending_texts) > MAX_EMBEDDING_INPUTS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"{len(pending_texts)} candidates require embedding, "
                f"which exceeds the {MAX_EMBEDDING_INPUTS} input limit"
            ),
        )
    pending_characters = sum(len(text) for text in pending_texts)
    if pending_characters > MAX_EMBEDDING_CHARS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"candidates requiring embedding total {pending_characters} characters, "
                f"which exceeds the {MAX_EMBEDDING_CHARS} limit"
            ),
        )

    try:
        # The raw query string is the retrieval cache key, so a rerank that
        # follows /query on the same query pays no query inference.
        query_vector = await _run(
            request, embedding_service.get_cached_query_embedding, body.query
        )
        candidate_vectors = (
            await _run(request, embedding_service.embed_texts, pending_texts)
            if pending_texts
            else []
        )
    except Exception as exc:
        logger.error(
            "Rerank embedding backend unavailable | candidates=%d pending=%d | %s: %s\n%s",
            len(body.candidates),
            len(pending_texts),
            type(exc).__name__,
            exc,
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Rerank profile '{body.profile}' backend is unavailable",
        )

    vectors: List[Optional[List[float]]] = [
        stored.get(candidate.id) for candidate in body.candidates
    ]
    for position, index in enumerate(pending):
        vectors[index] = candidate_vectors[position]

    similarities = [
        cosine_similarity(query_vector, vector) if vector else None
        for vector in vectors
    ]
    base_scores = [candidate.base_score for candidate in body.candidates]
    scores = blend_scores(similarities, base_scores, BlendConfig.from_env())
    ranked = order_by_score(scores, body.resolved_top_n())

    logger.info(
        "Reranked | profile=%s query=%s candidates=%d stored_vectors=%d embedded=%d returned=%d",
        body.profile,
        fingerprint(body.query),
        len(body.candidates),
        len(body.candidates) - len(pending_texts),
        len(pending_texts),
        len(ranked),
    )

    return RerankResponse(
        profile=body.profile,
        model=embedding_service.STORE_EMBEDDING_MODEL,
        results=[
            RerankResult(
                id=body.candidates[item.index].id,
                index=item.index,
                score=item.score,
            )
            for item in ranked
        ],
    )
