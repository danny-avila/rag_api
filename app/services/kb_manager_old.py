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
        """Create new knowledge base tables"""
        logger.info("Inside KB_manager service")
        if not KBManager.validate_kb_id(kb_id):
            logger.info("checking if the kb_id is valid or not.")
            raise ValueError(f"Invalid KB ID format: {kb_id}")

        try:
            # Get vector store instance (this will create tables if they don't exist)
            logger.info("Checking what get_vector_store does")
            vector_store = VectorStoreManager.get_vector_store(
                kb_id=kb_id,
                connection_string=CONNECTION_STRING,
                embeddings=embeddings,
                mode="async",
            )

            # Verify tables were created by checking collection exists
            collection_name = f"kb_{kb_id}"
            pool = await PSQLDatabase.get_pool()
            async with pool.acquire() as conn:
                result = await conn.fetchrow(
                    "SELECT uuid FROM langchain_pg_collection WHERE name = $1",
                    collection_name,
                )
                print("Result of creating table")

            if result:
                logger.info(f"KB created successfully: {kb_id}")
                return {
                    "kb_id": kb_id,
                    "collection_name": collection_name,
                    "status": "created",
                }
            else:
                raise Exception("Failed to create KB tables")

        except Exception as e:
            logger.error(f"Failed to create KB {kb_id}: {str(e)}")
            raise

    @staticmethod
    async def delete_kb(kb_id: str) -> dict:
        """Delete knowledge base and all its data"""
        try:
            collection_name = f"kb_{kb_id}"

            pool = await PSQLDatabase.get_pool()
            async with pool.acquire() as conn:
                # Delete embeddings first (FK constraint)
                await conn.execute(
                    """DELETE FROM langchain_pg_embedding 
                       WHERE collection_id IN (
                           SELECT uuid FROM langchain_pg_collection 
                           WHERE name = $1
                       )""",
                    collection_name,
                )

                # Delete collection
                result = await conn.execute(
                    "DELETE FROM langchain_pg_collection WHERE name = $1",
                    collection_name,
                )

            # Remove from cache
            VectorStoreManager.remove_vector_store(kb_id)

            logger.info(f"KB deleted successfully: {kb_id}")
            return {"kb_id": kb_id, "status": "deleted"}

        except Exception as e:
            logger.error(f"Failed to delete KB {kb_id}: {str(e)}")
            raise

    @staticmethod
    async def get_kb_info(kb_ids: List[str]) -> List[dict]:
        """Get information about multiple KBs"""
        try:
            pool = await PSQLDatabase.get_pool()
            async with pool.acquire() as conn:
                collection_names = [f"kb_{kb_id}" for kb_id in kb_ids]

                results = await conn.fetch(
                    """
                    SELECT 
                        c.name as collection_name,
                        c.uuid as collection_id,
                        COUNT(e.uuid) as document_count
                    FROM langchain_pg_collection c
                    LEFT JOIN langchain_pg_embedding e ON c.uuid = e.collection_id
                    WHERE c.name = ANY($1)
                    GROUP BY c.name, c.uuid
                """,
                    collection_names,
                )

                return [
                    {
                        "kb_id": row["collection_name"].replace("kb_", ""),
                        "collection_id": str(row["collection_id"]),
                        "document_count": row["document_count"],
                    }
                    for row in results
                ]

        except Exception as e:
            logger.error(f"Failed to get KB info: {str(e)}")
            raise
