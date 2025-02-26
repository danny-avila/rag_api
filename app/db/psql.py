# psql_helpers.py
import asyncpg
from app.config import DSN, logger

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

async def check_index_exists(conn, index_name: str) -> bool:
    result = await conn.fetchval("""
        SELECT EXISTS (
            SELECT FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = $1 AND n.nspname = 'public'
        );
    """, index_name)
    return result

async def ensure_custom_id_index_on_embedding():
    table_name = "langchain_pg_embedding"
    column_name = "custom_id"
    index_name = f"idx_{table_name}_{column_name}"
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        index_exists = await check_index_exists(conn, index_name)
        if not index_exists:
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({column_name});
            """)
            logger.debug(f"Created index '{index_name}' on '{table_name}({column_name})'")
        else:
            logger.debug(f"Index '{index_name}' already exists on '{table_name}({column_name})'")

async def ensure_jsonb_metadata():
    """
    Checks the data type of the 'cmetadata' column in 'langchain_pg_embedding' table.
    If the column is of type JSON, converts it to JSONB.
    """
    table_name = "langchain_pg_embedding"
    column_name = "cmetadata"
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        current_type = await conn.fetchval("""
            SELECT data_type
            FROM information_schema.columns
            WHERE table_name = $1 AND column_name = $2;
        """, table_name, column_name)
        if current_type == "json":
            await conn.execute(f"""
                ALTER TABLE {table_name}
                ALTER COLUMN {column_name}
                TYPE JSONB
                USING {column_name}::JSONB;
            """)
            logger.info(f"Converted column '{column_name}' in table '{table_name}' to JSONB")
        else:
            logger.info(f"Column '{column_name}' in table '{table_name}' is already JSONB or not of type JSON")

async def pg_health_check() -> bool:
    try:
        pool = await PSQLDatabase.get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False