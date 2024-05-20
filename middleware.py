import os
from datetime import datetime, timezone
from fastapi import Request
from fastapi.responses import JSONResponse
from jose import jwt, JWTError
from config import logger


async def security_middleware(request: Request, call_next):
    async def next():
        response = await call_next(request)
        return response

    if (request.url.path == "/docs" or
            request.url.path == "/openapi.json" or
            request.url.path == "/health"):
        return await next()

    jwt_secret = os.getenv('JWT_SECRET')

    if jwt_secret:
        authorization = request.headers.get('Authorization')
        if not authorization or not authorization.startswith('Bearer '):
            logger.info(f"Unauthorized request with missing or invalid Authorization header to: {request.url.path}")
            return JSONResponse(status_code=401, content = { "detail" : "Missing or invalid Authorization header" })

        token = authorization.split(' ')[1]
        try:
            payload = jwt.decode(token, jwt_secret, algorithms=['HS256'])

            # Check if the token has expired
            exp_timestamp = payload.get('exp')
            if exp_timestamp:
                exp_datetime = datetime.fromtimestamp(exp_timestamp, tz=timezone.utc)
                current_datetime = datetime.now(tz=timezone.utc)
                if current_datetime > exp_datetime:
                    logger.info(f"Unauthorized request with expired token to: {request.url.path}")
                    return JSONResponse(status_code=401, content = { "detail" : "Token has expired" })

            request.state.user = payload
            logger.debug(f"{request.url.path} - {payload}")
        except JWTError as e:
            logger.info(f"Unauthorized request with invalid token to: {request.url.path}, reason: {str(e)}")
            return JSONResponse(status_code=401, content = { "detail" : f"Invalid token: {str(e)}" })            
    else:
        logger.warn("JWT_SECRET not found in environment variables")

    return await next()
