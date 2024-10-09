from typing import Optional
from langchain_core.embeddings import Embeddings
from store import AsyncPgVector, ExtendedPCVector, ExtendedPgVector
from store import AtlasMongoVector
from pymongo import MongoClient
from pinecone import Pinecone 
from pinecone import ServerlessSpec
import time

def get_vector_store(
    connection_string: str,
    embeddings: Embeddings,
    collection_name: str,
    mode: str = "sync",
    search_index: Optional[str] = None,
    api_key: Optional[str]  = None
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
        return AtlasMongoVector(
            collection=mong_collection, embedding=embeddings, index_name=search_index
        )
    elif mode == "pinecone":
        region = connection_string
        index_name = collection_name
        pc = Pinecone(api_key)
        spec = ServerlessSpec(cloud="aws", region=region)
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
        if index_name not in existing_indexes:
            pc.create_index(
            name=index_name,
            dimension=get_dimension_size(embeddings),
            metric="cosine",
            spec=spec,
    )
            while not pc.describe_index(index_name).status["ready"]:
                time.sleep(1)
        host = pc.describe_index(index_name).host
        index = pc.Index(host=host)
        return ExtendedPCVector(index=index, embedding=embeddings)
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

def get_dimension_size(embeddings:Embeddings):
    return len(embeddings.embed_query("Dimensions"))