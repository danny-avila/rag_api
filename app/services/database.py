# app/services/database.py
import asyncpg
from app.config import DSN, logger


class PSQLDatabase:
    pool = None

    @classmethod
    async def get_pool(cls):
        if cls.pool is None:
            cls.pool = await asyncpg.create_pool(
                dsn=DSN,
                min_size=5,        # Minimum number of connections
                max_size=20,       # Maximum number of connections
                max_queries=1000,  # Maximum queries per connection
                max_inactive_connection_lifetime=300,  # 5 minutes
                command_timeout=60.0,  # 60 seconds command timeout
            )
            logger.info(f"Database pool created with min_size=5, max_size=20")
        return cls.pool

    @classmethod
    async def close_pool(cls):
        if cls.pool is not None:
            await cls.pool.close()
            cls.pool = None


async def ensure_vector_indexes():
    table_name = "langchain_pg_embedding"
    column_name = "custom_id"
    # You might want to standardize the index naming convention
    index_name = f"idx_{table_name}_{column_name}"

    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({column_name});
        """
        )

        await conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_file_id 
            ON {table_name} ((cmetadata->>'file_id'));
        """
        )

        logger.info("Vector database indexes ensured")


async def pg_health_check() -> bool:
    try:
        pool = await PSQLDatabase.get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False
