from typing import Optional
from functools import lru_cache
from pymongo import MongoClient
from langchain_core.embeddings import Embeddings

from .async_pg_vector import AsyncPgVector
from .atlas_mongo_vector import AtlasMongoVector
from .extended_pg_vector import ExtendedPgVector


@lru_cache(maxsize=4)
def _get_mongo_client(connection_string: str) -> MongoClient:
    """
    Get or create a cached MongoClient for the given connection string.

    Caches up to 4 clients to avoid creating new connections on every request,
    which prevents socket/memory leaks from accumulating MongoClient instances.

    :param connection_string: MongoDB connection URI
    :return: Cached MongoClient instance
    """
    return MongoClient(connection_string)


def get_vector_store(
    connection_string: str,
    embeddings: Embeddings,
    collection_name: str,
    mode: str = "sync",
    search_index: Optional[str] = None
):
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
        mongo_db = _get_mongo_client(connection_string).get_database()
        mong_collection = mongo_db[collection_name]
        return AtlasMongoVector(
            collection=mong_collection, embedding=embeddings, index_name=search_index
        )
    else:
        raise ValueError("Invalid mode specified. Choose 'sync', 'async', or 'atlas-mongo'.")