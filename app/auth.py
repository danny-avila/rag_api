"""JWT verification for rag_api.

Two token generations are recognised:

* **Strict** tokens are signed with ``RAG_JWT_SECRET`` — a dedicated key that
  must never be the LibreChat application ``JWT_SECRET`` — and carry issuer,
  audience, subject, tenant and scopes.
* **Legacy** tokens are the ``{"id": userId}`` shape LibreChat mints today.
  They are accepted only while ``RAG_AUTH_ACCEPT_LEGACY`` is true.

Key separation is what closes the bidirectional token-confusion channel: a
LibreChat session token (signed with ``JWT_SECRET``) is never accepted on the
``/v1`` service endpoints, in any mode. ``RAG_AUTH_ACCEPT_LEGACY`` relaxes the
*claim shape*, never the *signing key*, for those endpoints.
"""

import os
import jwt
from jwt import PyJWTError
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence

from fastapi import HTTPException, Request, status

from app.config import logger

SCOPE_EMBED = "rag:embed"
SCOPE_RERANK = "rag:rerank"
SCOPE_DOCUMENTS = "rag:documents"

BASE_TENANT_ID = "__BASE__"
SYSTEM_TENANT_ID = "__SYSTEM__"

SERVICE_PATH_PREFIX = "/v1/"
PUBLIC_PATHS = frozenset({"/docs", "/openapi.json", "/health"})

_HMAC_ALGORITHMS = frozenset({"HS256", "HS384", "HS512"})
_ASYMMETRIC_ALGORITHMS = frozenset(
    {"RS256", "RS384", "RS512", "ES256", "ES384", "ES512", "EdDSA"}
)
_SUPPORTED_ALGORITHMS = _HMAC_ALGORITHMS | _ASYMMETRIC_ALGORITHMS

_MIN_HMAC_SECRET_LENGTH = 32
_LEGACY_ALGORITHMS = ["HS256"]

# The subject claim of the pre-scopes ``{"id": userId}`` token shape, and the
# claims a token uses to state its own authorization. A token carrying either of
# the latter is never grandfathered into full privileges.
_LEGACY_SUBJECT_CLAIM = "id"
_AUTHORIZATION_CLAIMS = ("scopes", "scope", "entities")


class AuthError(Exception):
    """Raised when a bearer token cannot be accepted."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _env_flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("true", "1", "yes", "on", "y")


def _split_scopes(claims: Dict[str, Any]) -> FrozenSet[str]:
    raw = claims.get("scopes", claims.get("scope"))
    if raw is None:
        return frozenset()
    if isinstance(raw, str):
        return frozenset(part for part in raw.split() if part)
    if isinstance(raw, (list, tuple, set)):
        return frozenset(str(part) for part in raw if str(part))
    return frozenset()


def _string_set(value: Any) -> FrozenSet[str]:
    if value is None:
        return frozenset()
    if isinstance(value, str):
        return frozenset({value}) if value else frozenset()
    if isinstance(value, (list, tuple, set)):
        return frozenset(str(item) for item in value if str(item))
    return frozenset()


@dataclass(frozen=True)
class Principal:
    """The authenticated caller, as resolved from a verified bearer token."""

    subject: str
    tenant: str
    scopes: FrozenSet[str] = field(default_factory=frozenset)
    entities: FrozenSet[str] = field(default_factory=frozenset)
    legacy: bool = False

    def has_scope(self, scope: str) -> bool:
        """Legacy tokens predate scopes and are grandfathered into all of them.

        This is precisely what ``RAG_AUTH_ACCEPT_LEGACY=false`` turns off.
        """
        if self.legacy:
            return True
        return scope in self.scopes

    def permits_entity(self, entity_id: str) -> bool:
        """Whether the caller may act on documents owned by ``entity_id``.

        A legacy token carries no entity list, so the caller's claim is taken at
        face value — the same trust the current implementation extends. Strict
        tokens must name the entity explicitly.
        """
        if entity_id == self.subject:
            return True
        if self.legacy:
            return True
        return entity_id in self.entities


@dataclass(frozen=True)
class AuthSettings:
    rag_secret: Optional[str]
    rag_public_key: Optional[str]
    legacy_secret: Optional[str]
    algorithm: str
    issuer: str
    audience: str
    accept_legacy: bool
    leeway: int
    search_api_enabled: bool

    @classmethod
    def from_env(cls) -> "AuthSettings":
        return cls(
            rag_secret=(os.getenv("RAG_JWT_SECRET") or None),
            rag_public_key=(os.getenv("RAG_JWT_PUBLIC_KEY") or None),
            legacy_secret=(os.getenv("JWT_SECRET") or None),
            algorithm=os.getenv("RAG_JWT_ALGORITHM", "HS256").strip().upper(),
            issuer=os.getenv("RAG_JWT_ISSUER", "librechat").strip(),
            audience=os.getenv("RAG_JWT_AUDIENCE", "rag_api").strip(),
            accept_legacy=_env_flag("RAG_AUTH_ACCEPT_LEGACY", "true"),
            leeway=int(os.getenv("RAG_JWT_LEEWAY_SECONDS", "0")),
            search_api_enabled=_env_flag("RAG_SEARCH_API_ENABLED", "false"),
        )

    @property
    def verification_key(self) -> Optional[str]:
        if self.algorithm in _HMAC_ALGORITHMS:
            return self.rag_secret
        return self.rag_public_key

    @property
    def has_any_key(self) -> bool:
        return bool(self.verification_key or self.legacy_secret)

    def validate(self) -> None:
        """Fail closed on unusable signing configuration.

        Raised at import time by ``main`` so a misconfigured deployment never
        reaches a state where it silently serves unauthenticated traffic.
        """
        if self.algorithm not in _SUPPORTED_ALGORITHMS:
            raise RuntimeError(
                f"RAG_JWT_ALGORITHM '{self.algorithm}' is not supported "
                f"(expected one of {sorted(_SUPPORTED_ALGORITHMS)})"
            )

        if (
            self.rag_secret
            and self.legacy_secret
            and self.rag_secret == self.legacy_secret
        ):
            raise RuntimeError(
                "RAG_JWT_SECRET must differ from JWT_SECRET: sharing the key makes "
                "every rag_api token a full LibreChat session token and vice versa"
            )

        if not self.accept_legacy and not self.verification_key:
            raise RuntimeError(
                "RAG_AUTH_ACCEPT_LEGACY=false requires a verification key "
                "(RAG_JWT_SECRET, or RAG_JWT_PUBLIC_KEY for asymmetric algorithms)"
            )

        if not self.verification_key:
            if not self.search_api_enabled:
                return
            raise RuntimeError(
                "RAG_SEARCH_API_ENABLED=true requires "
                + (
                    "RAG_JWT_SECRET"
                    if self.algorithm in _HMAC_ALGORITHMS
                    else "RAG_JWT_PUBLIC_KEY"
                )
            )

        # Everything below validates a key that *is* configured, so it runs
        # whether or not the search router is mounted. The middleware accepts
        # RAG-signed strict tokens on /query, the upload routes and every other
        # non-/v1 path regardless of RAG_SEARCH_API_ENABLED, so gating these
        # checks on that flag would let a deployment protect its document APIs
        # with a trivially weak secret and still start cleanly.
        if (
            self.algorithm in _HMAC_ALGORITHMS
            and len(self.rag_secret) < _MIN_HMAC_SECRET_LENGTH
        ):
            raise RuntimeError(
                f"RAG_JWT_SECRET must be at least {_MIN_HMAC_SECRET_LENGTH} characters "
                f"for {self.algorithm}"
            )

        if not self.issuer:
            raise RuntimeError(
                "RAG_JWT_ISSUER must not be empty when a RAG signing key is configured"
            )

        if not self.audience:
            raise RuntimeError(
                "RAG_JWT_AUDIENCE must not be empty when a RAG signing key is configured"
            )


_settings: Optional[AuthSettings] = None


def get_settings() -> AuthSettings:
    global _settings
    if _settings is None:
        _settings = AuthSettings.from_env()
    return _settings


def reset_settings() -> None:
    """Drop the cached settings so the next read re-reads the environment."""
    global _settings
    _settings = None


def validate_startup_config() -> AuthSettings:
    settings = get_settings()
    settings.validate()
    if settings.search_api_enabled:
        logger.info(
            "Search API enabled | issuer=%s audience=%s algorithm=%s accept_legacy=%s",
            settings.issuer,
            settings.audience,
            settings.algorithm,
            settings.accept_legacy,
        )
    if settings.accept_legacy:
        logger.warning(
            "RAG_AUTH_ACCEPT_LEGACY=true: legacy tokens carry no entity list, so a "
            "caller-supplied entity_id is taken at face value. Set it false once "
            "every caller mints the full claim set."
        )
    return settings


def _principal_from_strict(claims: Dict[str, Any]) -> Principal:
    subject = str(claims.get("sub") or "")
    if not subject:
        raise AuthError(status.HTTP_401_UNAUTHORIZED, "Token is missing a subject")

    tenant = claims.get("tenant") or claims.get("tenant_id")
    if not tenant:
        raise AuthError(status.HTTP_401_UNAUTHORIZED, "Token is missing a tenant claim")
    tenant = str(tenant)

    scopes = _split_scopes(claims)
    if not scopes:
        raise AuthError(status.HTTP_401_UNAUTHORIZED, "Token carries no scopes")

    return Principal(
        subject=subject,
        tenant=tenant,
        scopes=scopes,
        entities=_string_set(claims.get("entities")),
        legacy=False,
    )


def _declares_authorization(claims: Dict[str, Any]) -> bool:
    """Whether the token states its own scopes or entity list."""
    return any(claim in claims for claim in _AUTHORIZATION_CLAIMS)


def _principal_from_legacy(claims: Dict[str, Any]) -> Principal:
    """Resolve a transition-era token, without widening what it asked for.

    The legacy grandfather in :meth:`Principal.has_scope` and
    :meth:`Principal.permits_entity` exists for the ``{"id": userId}`` shape,
    which predates scopes and entity lists entirely. It is withheld from every
    other token, because handing it to a token that *states* its authorization
    would mean a strict token gains privileges by being malformed: drop ``exp``,
    the tenant, or the scopes and a ``rag:embed``-only token would come back
    holding every scope and arbitrary entity access.

    So legacy acceptance covers claims a token omits, never claims it makes.
    """
    subject = str(claims.get("id") or claims.get("sub") or "")
    if not subject:
        raise AuthError(status.HTTP_401_UNAUTHORIZED, "Token is missing a subject")

    declares_authorization = _declares_authorization(claims)
    if declares_authorization:
        logger.warning(
            "Token failed strict validation but states its own authorization; "
            "honouring its scopes and entities rather than grandfathering it | "
            "subject=%s",
            subject,
        )

    return Principal(
        subject=subject,
        tenant=str(claims.get("tenant") or claims.get("tenant_id") or BASE_TENANT_ID),
        scopes=_split_scopes(claims),
        entities=_string_set(claims.get("entities")),
        legacy=_LEGACY_SUBJECT_CLAIM in claims and not declares_authorization,
    )


def _decode(
    token: str,
    key: str,
    algorithms: Sequence[str],
    settings: AuthSettings,
    strict: bool,
) -> Dict[str, Any]:
    if strict:
        return jwt.decode(
            token,
            key,
            algorithms=list(algorithms),
            audience=settings.audience,
            issuer=settings.issuer,
            leeway=settings.leeway,
            options={"require": ["exp", "iss", "aud", "sub"], "verify_aud": True},
        )
    return jwt.decode(
        token,
        key,
        algorithms=list(algorithms),
        leeway=settings.leeway,
        options={"verify_aud": False},
    )


def _legacy_claims_are_consistent(
    claims: Dict[str, Any], settings: AuthSettings
) -> bool:
    """Legacy acceptance covers *absent* claims, never *wrong* ones."""
    issuer = claims.get("iss")
    if issuer is not None and str(issuer) != settings.issuer:
        return False
    audience = claims.get("aud")
    if audience is None:
        return True
    audiences = audience if isinstance(audience, (list, tuple)) else [audience]
    return settings.audience in [str(item) for item in audiences]


def verify_token(
    token: str,
    settings: AuthSettings,
    *,
    allow_legacy_secret: bool,
) -> Principal:
    """Resolve a bearer token to a :class:`Principal` or raise :class:`AuthError`.

    ``allow_legacy_secret`` is false for the ``/v1`` service endpoints: those
    never accept a token signed with the LibreChat application secret, so a
    captured session token cannot reach them.
    """
    verification_key = settings.verification_key

    if verification_key:
        try:
            return _principal_from_strict(
                _decode(token, verification_key, [settings.algorithm], settings, True)
            )
        except jwt.ExpiredSignatureError:
            raise AuthError(status.HTTP_401_UNAUTHORIZED, "Token has expired")
        except (
            jwt.InvalidSignatureError,
            jwt.DecodeError,
            jwt.InvalidAlgorithmError,
        ):
            pass
        except (
            jwt.InvalidAudienceError,
            jwt.InvalidIssuerError,
            jwt.MissingRequiredClaimError,
            AuthError,
        ) as exc:
            # Signature verified but the claim set is not the strict one. That is
            # exactly the transition case, and nothing else.
            if not settings.accept_legacy:
                detail = exc.detail if isinstance(exc, AuthError) else str(exc)
                raise AuthError(
                    status.HTTP_401_UNAUTHORIZED, f"Invalid token: {detail}"
                )
        except PyJWTError as exc:
            raise AuthError(status.HTTP_401_UNAUTHORIZED, f"Invalid token: {exc}")

    if not settings.accept_legacy:
        raise AuthError(status.HTTP_401_UNAUTHORIZED, "Invalid token")

    candidate_keys: List[tuple] = []
    if verification_key:
        candidate_keys.append((verification_key, [settings.algorithm]))
    if allow_legacy_secret and settings.legacy_secret:
        candidate_keys.append((settings.legacy_secret, _LEGACY_ALGORITHMS))

    for key, algorithms in candidate_keys:
        try:
            claims = _decode(token, key, algorithms, settings, False)
        except jwt.ExpiredSignatureError:
            raise AuthError(status.HTTP_401_UNAUTHORIZED, "Token has expired")
        except PyJWTError:
            continue
        if not _legacy_claims_are_consistent(claims, settings):
            continue
        return _principal_from_legacy(claims)

    raise AuthError(status.HTTP_401_UNAUTHORIZED, "Invalid token")


def get_principal(request: Request) -> Principal:
    """FastAPI dependency: the principal the middleware already verified."""
    principal = getattr(request.state, "principal", None)
    if principal is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    return principal


def _refuse_without_scope(principal: Principal, scope: str) -> None:
    if principal.has_scope(scope):
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=f"Token is missing the '{scope}' scope",
    )


def require_scope(scope: str):
    """Dependency factory guarding a service endpoint with a single scope."""

    async def dependency(request: Request) -> Principal:
        settings = get_settings()
        if not settings.search_api_enabled:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Search API is not enabled on this deployment",
            )
        principal = get_principal(request)
        if principal.tenant == SYSTEM_TENANT_ID:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="System tenant may not call the search API",
            )
        _refuse_without_scope(principal, scope)
        return principal

    return dependency


async def require_document_scope(request: Request) -> Optional[Principal]:
    """Dependency guarding the file-addressed document routes.

    The document plane and the ``/v1`` inference plane are separate
    capabilities, so they carry separate scopes. ``rag:documents`` reads and
    deletes stored chunks; it buys no inference. ``rag:embed`` and
    ``rag:rerank`` spend inference budget; neither one substitutes for this.
    A credential minted to delete a file therefore cannot be replayed against
    an embedding provider, and vice versa.

    This is deliberately not :func:`require_scope`: that one gates the ``/v1``
    router and 503s when the search API is off, while these routes serve every
    deployment. A deployment with no signing key configured at all has no
    principal to check — the middleware already warns that such requests are
    unauthenticated — so scope enforcement starts where tokens do.
    """
    principal: Optional[Principal] = getattr(request.state, "principal", None)
    if principal is None:
        return None
    _refuse_without_scope(principal, SCOPE_DOCUMENTS)
    return principal
