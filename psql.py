# db.py
import asyncpg
from config import DSN, logger


class PSQLDatabase:
    pool = None

    @classmethod
    async def get_pool(cls):
        if cls.pool is None:
            cls.pool = await asyncpg.create_pool(dsn=DSN)
        return cls.pool

    @classmethod
    async def close_pool(cls):
        if cls.pool is not None:
            await cls.pool.close()
            cls.pool = None


async def ensure_custom_id_index_on_embedding():
    table_name = "langchain_pg_embedding"
    column_name = "custom_id"
    # You might want to standardize the index naming convention
    index_name = f"idx_{table_name}_{column_name}"

    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        # Check if the index exists
        index_exists = await check_index_exists(conn, index_name)

        if not index_exists:
            # If the index does not exist, create it
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({column_name});
            """)
            logger.debug(f"Created index '{index_name}' on '{table_name}({column_name})'")
        else:
            logger.debug(f"Index '{index_name}' already exists on '{table_name}({column_name})'")


async def check_index_exists(conn, index_name: str) -> bool:
    # Adjust the SQL query if necessary
    result = await conn.fetchval("""
        SELECT EXISTS (
            SELECT FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE  c.relname = $1 AND n.nspname = 'public' -- Adjust schema if necessary
        );
    """, index_name)
    return result


async def pg_health_check() -> bool:
    try:
        pool = await PSQLDatabase.get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False
