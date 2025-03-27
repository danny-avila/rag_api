# app/middleware.py
import os
import jwt
from jwt import PyJWTError
from fastapi import Request
from datetime import datetime, timezone
from fastapi.responses import JSONResponse
from app.config import logger


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