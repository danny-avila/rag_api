from typing import Optional, Dict, Any
from pymongo import MongoClient
from langchain_core.embeddings import Embeddings
from threading import Lock

from .async_pg_vector import AsyncPgVector
from .atlas_mongo_vector import AtlasMongoVector
from .extended_pg_vector import ExtendedPgVector


def get_vector_store(
    connection_string: str,
    embeddings: Embeddings,
    collection_name: str,
    mode: str = "sync",
    search_index: Optional[str] = None,
):
    from app.config import logger

    logger.info("factory get vector store function")
    if mode == "sync":
        print("synchronous connection")
        return ExtendedPgVector(
            connection_string=connection_string,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
    elif mode == "async":
        print("Asynchronous connection")
        return AsyncPgVector(
            connection_string=connection_string,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
    elif mode == "atlas-mongo":
        print("Mongo Atlas connection")
        mongo_db = MongoClient(connection_string).get_database()
        mong_collection = mongo_db[collection_name]
        return AtlasMongoVector(
            collection=mong_collection, embedding=embeddings, index_name=search_index
        )
    else:
        raise ValueError(
            "Invalid mode specified. Choose 'sync', 'async', or 'atlas-mongo'."
        )


class VectorStoreManager:
    _instances: Dict[str, Any] = {}
    _lock = Lock()

    @classmethod
    def get_vector_store(
        cls,
        kb_id: str,
        connection_string: str,
        embeddings: Embeddings,
        mode: str = "async",
    ):
        """Get or create vector store instance for a KB"""
        from app.config import logger

        with cls._lock:
            if kb_id not in cls._instances:
                logger.info(
                    "vector client instance for the Knowledge base not found, creating new."
                )
                collection_name = f"collection_{kb_id}"
                logger.info("collection name ", collection_name)
                cls._instances[kb_id] = get_vector_store(
                    connection_string=connection_string,
                    embeddings=embeddings,
                    collection_name=collection_name,
                    mode=mode,
                )
                logger.info(f"Created new vector store instance for KB: {kb_id}")
            return cls._instances[kb_id]

    @classmethod
    def remove_vector_store(cls, kb_id: str):
        """Remove vector store instance from cache"""
        from app.config import logger

        with cls._lock:
            if kb_id in cls._instances:
                del cls._instances[kb_id]
                logger.info(f"Removed vector store instance for KB: {kb_id}")
