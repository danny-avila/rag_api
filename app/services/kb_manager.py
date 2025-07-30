import uuid
import re
from typing import List, Optional
from app.config import CONNECTION_STRING, embeddings, logger
from app.services.database import PSQLDatabase
from app.services.vector_store.factory import VectorStoreManager


class KBManager:
    @staticmethod
    def validate_kb_id(kb_id: str) -> bool:
        """Validate KB ID format (kb_<uuid_without_hyphens>)"""
        pattern = r"^kb_[a-f0-9]{32}$"
        return bool(re.match(pattern, kb_id))

    @staticmethod
    async def create_kb(kb_id: str) -> dict:
        """Create new knowledge base with dedicated tables"""
        if not KBManager.validate_kb_id(kb_id):
            raise ValueError(f"Invalid KB ID format: {kb_id}")

        collection_table = f"collection_{kb_id}"
        embedding_table = f"embedding_{kb_id}"

        try:
            # Create KB-specific vector store with dedicated tables
            pool = await PSQLDatabase.get_pool()
            async with pool.acquire() as conn:
                # Create KB-specific tables.
                await conn.execute(
                    f"""
                    CREATE TABLE {collection_table} (
                        uuid UUID DEFAULT gen_random_uuid() PRIMARY KEY,
                        name VARCHAR NOT NULL,
                        cmetadata JSONB
                    );
                    
                    CREATE TABLE {embedding_table}(
                        uuid UUID DEFAULT gen_random_uuid() PRIMARY KEY,
                        collection_id UUID REFERENCES {collection_table}(uuid) ON DELETE CASCADE,
                        embedding VECTOR(1536),
                        document TEXT,
                        cmetadata JSONB,
                        custom_id VARCHAR
                    );
                    
                    -- KB-specific indexes
                    CREATE INDEX ix_{kb_id}_collection_name ON {collection_table} (name);
                    CREATE INDEX ix_{kb_id}_embedding_collection_id ON {embedding_table} (collection_id);
                    CREATE INDEX ix_{kb_id}_embedding_custom_id ON {embedding_table} (custom_id);
                    
                    -- KB-specific HNSW index (this is the key!)
                    CREATE INDEX ix_{kb_id}_embedding_vector 
                    ON {embedding_table} 
                    USING hnsw (embedding vector_l2_ops);
                """
                )

            logger.info(f"KB created successfully: {kb_id}")
            return {"kb_id": kb_id, "status": "created"}

        except Exception as e:
            logger.error(f"Failed to create KB {kb_id}: {str(e)}")
            raise

    @staticmethod
    async def delete_kb(kb_id: str) -> dict:
        """Delete knowledge base and all its dedicated tables"""
        try:
            collection_table = f"kb_{kb_id}_collection"
            embedding_table = f"kb_{kb_id}_embedding"

            pool = await PSQLDatabase.get_pool()
            async with pool.acquire() as conn:
                # Drop KB-specific tables (CASCADE handles FK constraints)
                await conn.execute(f"DROP TABLE IF EXISTS {embedding_table} CASCADE")
                await conn.execute(f"DROP TABLE IF EXISTS {collection_table} CASCADE")

                logger.info(f"Dropped dedicated tables for KB: {kb_id}")

            # Remove from cache
            VectorStoreManager.remove_vector_store(kb_id)

            logger.info(f"KB deleted successfully: {kb_id}")
            return {"kb_id": kb_id, "status": "deleted"}

        except Exception as e:
            logger.error(f"Failed to delete KB {kb_id}: {str(e)}")
            raise

    @staticmethod
    async def get_kb_info(kb_ids: List[str]) -> List[dict]:
        """Get information about multiple KBs from their dedicated tables"""
        try:
            results = []

            for kb_id in kb_ids:
                collection_table = f"kb_{kb_id}_collection"
                embedding_table = f"kb_{kb_id}_embedding"

                pool = await PSQLDatabase.get_pool()
                async with pool.acquire() as conn:
                    # Check if KB tables exist
                    table_exists = await conn.fetchrow(
                        "SELECT tablename FROM pg_tables WHERE tablename = $1",
                        collection_table,
                    )

                    if table_exists:
                        # Count documents in KB-specific table
                        count_result = await conn.fetchrow(
                            f"SELECT COUNT(*) as document_count FROM {embedding_table}"
                        )

                        # Get collection info from KB-specific table
                        collection_result = await conn.fetchrow(
                            f"SELECT uuid FROM {collection_table} LIMIT 1"
                        )

                        results.append(
                            {
                                "kb_id": kb_id,
                                "collection_id": (
                                    str(collection_result["uuid"])
                                    if collection_result
                                    else None
                                ),
                                "document_count": count_result["document_count"],
                                "has_dedicated_tables": True,
                            }
                        )
                    else:
                        results.append(
                            {
                                "kb_id": kb_id,
                                "collection_id": None,
                                "document_count": 0,
                                "has_dedicated_tables": False,
                                "status": "not_found",
                            }
                        )

            return results

        except Exception as e:
            logger.error(f"Failed to get KB info: {str(e)}")
            raise
