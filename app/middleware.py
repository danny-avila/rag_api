# app/middleware.py
from fastapi import Request
from fastapi.responses import JSONResponse

from app.config import logger
from app.auth import (
    AuthError,
    PUBLIC_PATHS,
    SERVICE_PATH_PREFIX,
    get_settings,
    verify_token,
)

_UNCONFIGURED_WARNING = (
    "Neither RAG_JWT_SECRET nor JWT_SECRET is set — requests are not authenticated"
)

_warned_unconfigured = False


def _warn_unconfigured_once() -> None:
    global _warned_unconfigured
    if not _warned_unconfigured:
        _warned_unconfigured = True
        logger.warning(_UNCONFIGURED_WARNING)


async def security_middleware(request: Request, call_next):
    async def next_middleware_call():
        return await call_next(request)

    path = request.url.path
    if path in PUBLIC_PATHS:
        return await next_middleware_call()

    settings = get_settings()
    is_service_path = path.startswith(SERVICE_PATH_PREFIX)

    if is_service_path and not settings.search_api_enabled:
        return JSONResponse(
            status_code=503,
            content={"detail": "Search API is not enabled on this deployment"},
        )

    if not settings.has_any_key:
        _warn_unconfigured_once()
        return await next_middleware_call()

    authorization = request.headers.get("Authorization")
    if not authorization or not authorization.startswith("Bearer "):
        logger.info(
            "Unauthorized request with missing or invalid Authorization header to: %s",
            path,
        )
        return JSONResponse(
            status_code=401,
            content={"detail": "Missing or invalid Authorization header"},
        )

    token = authorization.split(" ", 1)[1]
    try:
        # The service endpoints never accept a token signed with the LibreChat
        # application secret, regardless of the legacy transition flag.
        principal = verify_token(
            token, settings, allow_legacy_secret=not is_service_path
        )
    except AuthError as exc:
        logger.info(
            "Unauthorized request to: %s, reason: %s",
            path,
            exc.detail,
        )
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    request.state.principal = principal
    request.state.user = {
        "id": principal.subject,
        "sub": principal.subject,
        "tenant": principal.tenant,
    }
    logger.debug(
        "%s - subject=%s tenant=%s legacy=%s",
        path,
        principal.subject,
        principal.tenant,
        principal.legacy,
    )

    return await next_middleware_call()
