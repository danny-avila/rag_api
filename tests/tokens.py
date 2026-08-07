"""Token minting helpers shared by the auth and service-endpoint tests."""

import datetime
from typing import Iterable, Optional

import jwt

# Both secrets are >= 32 chars so they satisfy the startup length check.
RAG_SECRET = "rag-signing-secret-0123456789abcdef"
APP_SECRET = "librechat-app-secret-0123456789abcdef"

ISSUER = "librechat"
AUDIENCE = "rag_api"

BASE_TENANT = "__BASE__"
SYSTEM_TENANT = "__SYSTEM__"


def _exp(seconds: int) -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
        seconds=seconds
    )


def strict_token(
    subject: str = "user-1",
    tenant: Optional[str] = BASE_TENANT,
    scopes: Optional[Iterable[str]] = ("rag:embed", "rag:rerank"),
    entities: Iterable[str] = (),
    secret: str = RAG_SECRET,
    issuer: Optional[str] = ISSUER,
    audience: Optional[str] = AUDIENCE,
    expires_in: int = 300,
    algorithm: str = "HS256",
) -> str:
    claims = {"sub": subject, "exp": _exp(expires_in)}
    if issuer is not None:
        claims["iss"] = issuer
    if audience is not None:
        claims["aud"] = audience
    if tenant is not None:
        claims["tenant"] = tenant
    if scopes is not None:
        claims["scopes"] = list(scopes)
    if entities:
        claims["entities"] = list(entities)
    return jwt.encode(claims, secret, algorithm=algorithm)


def legacy_token(
    user_id: str = "user-1",
    secret: str = APP_SECRET,
    expires_in: int = 300,
) -> str:
    """The ``{ id }`` shape LibreChat's generateShortLivedToken mints today."""
    return jwt.encode(
        {"id": user_id, "exp": _exp(expires_in)}, secret, algorithm="HS256"
    )


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}
