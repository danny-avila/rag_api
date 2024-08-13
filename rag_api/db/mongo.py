import logging

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from rag_api.config import ATLAS_MONGO_DB_URI

logger = logging.getLogger(__name__)


async def mongo_health_check() -> bool:
    try:
        client = MongoClient(ATLAS_MONGO_DB_URI)
        client.admin.command("ping")
        return True
    except PyMongoError as e:
        logger.error(f"MongoDB health check failed: {e}")
        return False
