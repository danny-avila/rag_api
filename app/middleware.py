import os
import jwt
import json
from jwt import PyJWTError
from datetime import datetime, timezone
from fastapi import Request
from fastapi.responses import JSONResponse
from app.config import logger

async def security_middleware(request: Request, call_next):
    if request.url.path in {"/docs", "/openapi.json", "/health"}:
        return await call_next(request)

    jwt_secret = os.getenv("JWT_SECRET")
    if not jwt_secret:
        logger.warning("JWT_SECRET not found in environment variables")
        return await call_next(request)

    authorization = request.headers.get("Authorization")
    if not authorization or not authorization.startswith("Bearer "):
        logger.info(f"Unauthorized request to: {request.url.path}")
        return JSONResponse(status_code=401, content={"detail": "Missing or invalid Authorization header"})

    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, jwt_secret, algorithms=["HS256"])
        exp_timestamp = payload.get("exp")
        if exp_timestamp and datetime.now(tz=timezone.utc) > datetime.fromtimestamp(exp_timestamp, tz=timezone.utc):
            logger.info(f"Unauthorized request with expired token to: {request.url.path}")
            return JSONResponse(status_code=401, content={"detail": "Token has expired"})
        request.state.user = payload
        logger.debug(f"{request.url.path} - {payload}")
    except PyJWTError as e:
        logger.info(f"Unauthorized request with invalid token to: {request.url.path}, reason: {str(e)}")
        return JSONResponse(status_code=401, content={"detail": f"Invalid token: {str(e)}"})

    return await call_next(request)

from starlette.middleware.base import BaseHTTPMiddleware

class LogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        logger_method = logger.info
        if str(request.url).endswith("/health"):
            logger_method = logger.debug
        logger_method(
            f"Request {request.method} {request.url} - {response.status_code}",
            extra={
                "http_req": {"method": request.method, "url": str(request.url)},
                "http_res": {"status_code": response.status_code},
            },
        )
        return response