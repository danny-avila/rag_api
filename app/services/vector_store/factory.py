import logging
from typing import Optional

from pymongo import MongoClient
from langchain_core.embeddings import Embeddings

from .async_pg_vector import AsyncPgVector
from .atlas_mongo_vector import AtlasMongoVector
from .extended_pg_vector import ExtendedPgVector

logger = logging.getLogger(__name__)

# Holds the MongoClient so it can be closed on shutdown.
_mongo_client: Optional[MongoClient] = None


def get_vector_store(
    connection_string: str,
    embeddings: Embeddings,
    collection_name: str,
    mode: str = "sync",
    search_index: Optional[str] = None,
):
    global _mongo_client

    if mode == "sync":
        return ExtendedPgVector(
            connection_string=connection_string,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
    elif mode == "async":
        return AsyncPgVector(
            connection_string=connection_string,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
    elif mode == "atlas-mongo":
        _mongo_client = MongoClient(connection_string)
        mongo_db = _mongo_client.get_database()
        mong_collection = mongo_db[collection_name]
        return AtlasMongoVector(
            collection=mong_collection, embedding=embeddings, index_name=search_index
        )
    else:
        raise ValueError(
            "Invalid mode specified. Choose 'sync', 'async', or 'atlas-mongo'."
        )


def close_vector_store_connections(vector_store) -> None:
    """Close connections held by the vector store and its backing clients."""
    global _mongo_client

    # Close MongoDB client if one was created
    if _mongo_client is not None:
        try:
            _mongo_client.close()
            logger.info("MongoDB client closed")
        except Exception as e:
            logger.warning("Failed to close MongoDB client: %s", e)
        finally:
            _mongo_client = None

    # Dispose SQLAlchemy engine if the vector store has one
    engine = getattr(vector_store, "_bind", None)
    if engine is not None and hasattr(engine, "dispose"):
        try:
            engine.dispose()
            logger.info("SQLAlchemy engine disposed")
        except Exception as e:
            logger.warning("Failed to dispose SQLAlchemy engine: %s", e)
