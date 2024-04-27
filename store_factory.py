from langchain_community.embeddings import OpenAIEmbeddings

from store import AsyncPgVector, ExtendedPgVector

from langchain_community.vectorstores import Qdrant

import qdrant_client


def get_vector_store(
    connection_string: str,
    embeddings: OpenAIEmbeddings,
    collection_name: str,
    mode: str = "sync",
    vector_db: str = None,
    qdrant_host: str = None,
    qdrant_api_key: str = None
):
    print(vector_db)
    if vector_db == "pgvector":
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
        else:
            raise ValueError("Invalid mode specified. Choose 'sync' or 'async'.")
    elif vector_db == "qdrant":
        print("Imprimindo embeddings", embeddings)

        client = qdrant_client.QdrantClient(
            qdrant_host,
            api_key=qdrant_api_key
        )

        collection_config = qdrant_client.http.models.VectorParams(
            size=768,  # Adjust as per your model's embedding size
            distance=qdrant_client.http.models.Distance.COSINE
        )

        # Recreate collection if needed
        try:
            # Verify if collection exists
            collection = client.get_collection(collection_name=collection_name)
        except Exception:
            # Recreate collection
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=collection_config
            )
    
        vector_store = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings
        )

        return vector_store

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