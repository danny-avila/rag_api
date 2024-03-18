import os
from datetime import datetime, timezone
from fastapi import Request, HTTPException
from jose import jwt, JWTError
from config import logger

async def security_middleware(request: Request, call_next):
    async def next():
        response = await call_next(request)
        return response

    if request.url.path == "/docs" or request.url.path == "/openapi.json":
        return await next()

    jwt_secret = os.getenv('JWT_SECRET')

    if jwt_secret:
        authorization = request.headers.get('Authorization')
        if not authorization or not authorization.startswith('Bearer '):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

        token = authorization.split(' ')[1]
        try:
            payload = jwt.decode(token, jwt_secret, algorithms=['HS256'])
            
            # Check if the token has expired
            exp_timestamp = payload.get('exp')
            if exp_timestamp:
                exp_datetime = datetime.fromtimestamp(exp_timestamp, tz=timezone.utc)
                current_datetime = datetime.now(tz=timezone.utc)
                if current_datetime > exp_datetime:
                    raise HTTPException(status_code=401, detail="Token has expired")
            
            request.state.user = payload
            logger.debug(f"{request.url.path} - {payload}")
        except JWTError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
    else:
        logger.warn("JWT_SECRET not found in environment variables")

    return await next()
