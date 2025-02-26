import traceback

from fastapi import APIRouter

from app.config import logger, VECTOR_DB_TYPE, VectorDBType
from app.db.mongo import mongo_health_check
from app.db.psql import pg_health_check

router = APIRouter()

def is_health_ok():
    if VECTOR_DB_TYPE == VectorDBType.PGVECTOR:
        return pg_health_check()
    if VECTOR_DB_TYPE == VectorDBType.ATLAS_MONGO:
        return mongo_health_check()
    else:
        return True


@router.get("/health")
async def health_check():
    try:
        if await is_health_ok():
            return {"status": "UP"}
        else:
            logger.error("Health check failed")
            return {"status": "DOWN"}, 503
    except Exception as e:
        logger.error(
            "Error during health check | Error: %s | Traceback: %s",
            str(e),
            traceback.format_exc(),
        )
        return {"status": "DOWN", "error": str(e)}, 503