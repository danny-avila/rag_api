# app/services/mongo_client.py
import asyncio
import logging
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from app.config import ATLAS_MONGO_DB_URI

logger = logging.getLogger(__name__)


async def mongo_health_check() -> bool:
    client = None
    try:
        client = await asyncio.to_thread(MongoClient, ATLAS_MONGO_DB_URI)
        await asyncio.to_thread(client.admin.command, "ping")
        return True
    except PyMongoError as e:
        logger.error("MongoDB health check failed: %s", e)
        return False
    finally:
        if client is not None:
            try:
                await asyncio.to_thread(client.close)
            except Exception as e:
                logger.debug("Failed to close health check client: %s", e)
