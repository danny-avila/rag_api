from pymongo import MongoClient
from pymongo.errors import PyMongoError
from app.config import ATLAS_MONGO_DB_URI, logger

async def mongo_health_check() -> bool:
    try:
        client = MongoClient(ATLAS_MONGO_DB_URI)
        client.admin.command("ping")
        return True
    except PyMongoError as e:
        logger.error(f"MongoDB health check failed: {e}")
        return False