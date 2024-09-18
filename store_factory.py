from typing import Optional, TypedDict
from langchain_community.embeddings import OpenAIEmbeddings
from pymongo import MongoClient
import qdrant_client
from store import AsyncPgVector, ExtendedPgVector, AsyncQdrant, AtlasMongoVector


class QdrantConfig(TypedDict, total=False):
    qdrant_host: Optional[str]
    qdrant_api_key:  Optional[str]
    qdrant_embeddings_dimension: Optional[int]

def get_vector_store(
    connection_string: str,
    embeddings: OpenAIEmbeddings,
    collection_name: str,
    mode: str = "sync",
    *,
    additional_kwargs: Optional[QdrantConfig]  = None 
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
        mongo_db = MongoClient(connection_string).get_database()
        mong_collection = mongo_db[collection_name]
        return AtlasMongoVector(collection=mong_collection, embedding=embeddings)    
    
    elif mode == "qdrant":
        if additional_kwargs is None:
            additional_kwargs = {}
        qdrant_host = additional_kwargs['qdrant_host']
        qdrant_api_key = additional_kwargs.get['qdrant_api_key']
        qdrant_embeddings_dimension = additional_kwargs.get('qdrant_embeddings_dimension')
        client = qdrant_client.QdrantClient(
        qdrant_host,
        api_key=qdrant_api_key
        )

        collection_config = qdrant_client.http.models.VectorParams(
            size=qdrant_embeddings_dimension,
            distance=qdrant_client.http.models.Distance.COSINE
        )

        print(f"Creating collection {collection_name}...")
        try:
            # Verify if collection exists
            collection = client.get_collection(collection_name=collection_name)
        except Exception:
            # Recreate collection
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=collection_config
            )
        return AsyncQdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings
            
            )
    
    elif mode == "atlas-mongo":
        mongo_db = MongoClient(connection_string).get_database()
        mong_collection = mongo_db[collection_name]
        return AtlasMongoVector(collection=mong_collection, embedding=embeddings, index_name=collection_name)

    else:
        raise ValueError("Invalid mode specified. Choose 'sync' or 'async'.")


async def create_index_if_not_exists(conn, table_name: str, column_name: str):
    # Construct index name conventionally
    index_name = f"idx_{table_name}_{column_name}"
    # Check if index exists
    exists = await conn.fetchval(f"""
        SELECT EXISTS (
            SELECT FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = $1
            AND n.nspname = 'public'  -- Or specify your schema if different
        );
    """, index_name)
    # Create the index if it does not exist
    if not exists:
        await conn.execute(f"""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS {index_name}
            ON public.{table_name} ({column_name});
        """)
        print(f"Index {index_name} created on {table_name}.{column_name}")
    else:
        print(f"Index {index_name} already exists on {table_name}.{column_name}")