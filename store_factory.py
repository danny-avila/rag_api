from typing import Optional, TypedDict
from langchain_core.embeddings import Embeddings
from store import AsyncPgVector, ExtendedPgVector
from store import AtlasMongoVector
from store import AsyncQdrant
import qdrant_client
from pymongo import MongoClient


async_DB = (AsyncPgVector, AsyncQdrant) #Add if async database implementation 


def get_vector_store(
    connection_string: str,
    embeddings: Embeddings,
    collection_name: str,
    mode: str = "sync-PGVector",
    search_index: Optional[str] = None,
    api_key: Optional[str]  = None 
):
    if mode == "sync-PGVector":
        return ExtendedPgVector(
            connection_string=connection_string,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
    elif mode == "async-PGVector":
        return AsyncPgVector(
            connection_string=connection_string,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
    elif mode == "atlas-mongo":
        mongo_db = MongoClient(connection_string).get_database()
        mong_collection = mongo_db[collection_name]
        return AtlasMongoVector(
            collection=mong_collection, embedding=embeddings, index_name=search_index
        )
    elif mode == "qdrant":
        embeddings_dimension = len(embeddings.embed_query("Dimension"))
        client = qdrant_client.QdrantClient(
        url=connection_string,
        api_key=api_key
        )
        collection_config = qdrant_client.http.models.VectorParams(
            size=embeddings_dimension,
            distance=qdrant_client.http.models.Distance.COSINE
        )
        if not client.collection_exists(collection_name):
            collection_config = qdrant_client.http.models.VectorParams(
                size=embeddings_dimension,
                distance=qdrant_client.http.models.Distance.COSINE
            )
            client.create_collection(
            collection_name=collection_name,
            vectors_config=collection_config
            )
        return AsyncQdrant(
            client=client,
            collection_name=collection_name,
            embedding=embeddings      
            )

    else:
        raise ValueError("Invalid mode specified. Choose 'sync' or 'async'.")


async def create_index_if_not_exists(conn, table_name: str, column_name: str):
    # Construct index name conventionally
    index_name = f"idx_{table_name}_{column_name}"
    # Check if index exists
    exists = await conn.fetchval(
        f"""
        SELECT EXISTS (
            SELECT FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = $1
            AND n.nspname = 'public'  -- Or specify your schema if different
        );
    """,
        index_name,
    )
    # Create the index if it does not exist
    if not exists:
        await conn.execute(
            f"""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS {index_name}
            ON public.{table_name} ({column_name});
        """
        )
        print(f"Index {index_name} created on {table_name}.{column_name}")
    else:
        print(f"Index {index_name} already exists on {table_name}.{column_name}")
