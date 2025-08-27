# app/middleware.py
import os
import jwt
import asyncio
from jwt import PyJWTError
from fastapi import Request, HTTPException
from datetime import datetime, timezone
from fastapi.responses import JSONResponse
from app.config import logger

# Global semaphore to limit concurrent embed requests  
# Configurable via environment variable, default to 3 concurrent requests
def get_embed_concurrency_limit():
    value = os.getenv("EMBED_CONCURRENCY_LIMIT", "3")
    # Strip comments and whitespace from environment variables
    if isinstance(value, str) and '#' in value:
        value = value.split('#')[0].strip()
    return int(value)

EMBED_CONCURRENCY_LIMIT = get_embed_concurrency_limit()
embed_semaphore = asyncio.Semaphore(EMBED_CONCURRENCY_LIMIT)

logger.info(f"Initialized embed throttling middleware with concurrency limit: {EMBED_CONCURRENCY_LIMIT}")


async def security_middleware(request: Request, call_next):
    async def next_middleware_call():
        return await call_next(request)

    if request.url.path in {"/docs", "/openapi.json", "/health"}:
        return await next_middleware_call()

    jwt_secret = os.getenv("JWT_SECRET")
    if not jwt_secret:
        logger.warn("JWT_SECRET not found in environment variables")
        return await next_middleware_call()

    authorization = request.headers.get("Authorization")
    if not authorization or not authorization.startswith("Bearer "):
        logger.info(
            f"Unauthorized request with missing or invalid Authorization header to: {request.url.path}"
        )
        return JSONResponse(
            status_code=401,
            content={"detail": "Missing or invalid Authorization header"},
        )

    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, jwt_secret, algorithms=["HS256"])
        exp_timestamp = payload.get("exp")
        if exp_timestamp and datetime.now(tz=timezone.utc) > datetime.fromtimestamp(
            exp_timestamp, tz=timezone.utc
        ):
            logger.info(
                f"Unauthorized request with expired token to: {request.url.path}"
            )
            return JSONResponse(
                status_code=401, content={"detail": "Token has expired"}
            )

        request.state.user = payload
        logger.debug(f"{request.url.path} - {payload}")
    except PyJWTError as e:
        logger.info(
            f"Unauthorized request with invalid token to: {request.url.path}, reason: {str(e)}"
        )
        return JSONResponse(
            status_code=401, content={"detail": f"Invalid token: {str(e)}"}
        )

    return await next_middleware_call()


async def throttle_embed_middleware(request: Request, call_next):
    """
    Middleware to throttle concurrent embed requests to prevent system overload.
    LibreChat sends many concurrent requests which can overwhelm the embedding/database systems.
    """
    # Only throttle embed endpoints
    if request.url.path in ["/embed", "/local/embed", "/embed-upload"]:
        logger.info(f"Acquiring embed semaphore for {request.url.path} (limit: {EMBED_CONCURRENCY_LIMIT})")
        async with embed_semaphore:
            logger.info(f"Processing embed request for {request.url.path}")
            response = await call_next(request)
            logger.info(f"Completed embed request for {request.url.path}")
            return response
    else:
        # No throttling for non-embed requests
        return await call_next(request)


async def timeout_middleware(request: Request, call_next):
    """
    Middleware to handle request timeouts gracefully and provide better error messages.
    """
    # Set different timeouts based on the endpoint
    if request.url.path == "/embed" or request.url.path == "/local/embed":
        timeout_seconds = 300  # 5 minutes for embedding operations
    elif request.url.path == "/query" or request.url.path == "/query_multiple":
        timeout_seconds = 60  # 1 minute for queries
    elif request.url.path == "/health":
        timeout_seconds = 10  # 10 seconds for health checks
    else:
        timeout_seconds = 120  # 2 minutes default
    
    try:
        # Execute the request with timeout
        response = await asyncio.wait_for(
            call_next(request), 
            timeout=timeout_seconds
        )
        return response
        
    except asyncio.TimeoutError:
        logger.error(
            f"Request timeout after {timeout_seconds}s for {request.method} {request.url.path}"
        )
        return JSONResponse(
            status_code=504,
            content={
                "detail": f"Request timed out after {timeout_seconds} seconds",
                "error": "timeout",
                "suggestion": "Try with smaller files or reduce batch size"
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error in timeout middleware: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "error": str(e)
            }
        )