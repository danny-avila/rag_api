from typing import Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from store import AsyncPgVector, ExtendedPgVector
from store import AtlasMongoVector
from pymongo import MongoClient


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
        mongo_db = MongoClient(connection_string).get_database()
        mong_collection = mongo_db[collection_name]
        return AtlasMongoVector(
            collection=mong_collection, embedding=embeddings, index_name=search_index
        )
    elif mode == "dummy":
        # Return a fake vector store that does nothing.
        class DummyVectorStore:
            def get_all_ids(self) -> list[str]:
                return []  # Or return dummy IDs if needed.

            def get_documents_by_ids(self, ids: list[str]) -> list[Document]:
                return []  # Return an empty list of documents.

            def delete(self, ids: Optional[list[str]] = None, collection_only: bool = False) -> None:
                pass  # No-op.

        return DummyVectorStore()
    else:
        raise ValueError("Invalid mode specified. Choose 'sync', 'async', 'atlas-mongo', or 'dummy'.")


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
