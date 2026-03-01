# app/services/database.py
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


async def ensure_vector_indexes():
    """Ensure required indexes on langchain_pg_embedding and migrate cmetadata to JSONB.

    Runs at startup. Idempotent — safe to call repeatedly.
    Operations:
      1. B-tree index on custom_id.
      2. Expression index on (cmetadata->>'file_id').
      3. DDL migration: JSON -> JSONB for cmetadata (skipped if already JSONB).
      4. GIN index (jsonb_path_ops) on cmetadata for containment queries.
    """
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

        # Expression index for (cmetadata->>'file_id') text queries.
        # NOTE: After the JSONB migration, LangChain generates @> containment
        # queries served by ix_cmetadata_gin instead. Consider dropping this
        # index in a follow-up once JSONB filtering is confirmed stable.
        await conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_file_id
            ON {table_name} ((cmetadata->>'file_id'));
        """
        )

        # Migrate cmetadata from JSON to JSONB (idempotent — skipped if already JSONB).
        # Rollback: ALTER TABLE langchain_pg_embedding ALTER COLUMN cmetadata TYPE JSON USING cmetadata::json;
        # NOTE: table name is hardcoded below (not interpolated) to avoid SQL injection.
        await conn.execute(
            """
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'langchain_pg_embedding'
                      AND table_schema = current_schema()
                      AND column_name = 'cmetadata'
                      AND data_type = 'json'
                ) THEN
                    SET LOCAL lock_timeout = '10s';
                    ALTER TABLE langchain_pg_embedding
                        ALTER COLUMN cmetadata TYPE JSONB USING cmetadata::jsonb;
                END IF;
            END
            $$;
            """
        )

        # GIN index on cmetadata for efficient JSONB filtering
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS ix_cmetadata_gin
            ON langchain_pg_embedding
            USING gin (cmetadata jsonb_path_ops);
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
