# app/services/database.py
import os

import asyncpg
from app.config import DSN, POSTGRES_SCHEMA, logger
from app.services.vector_store.factory import _build_search_path, _parse_schemas


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "y", "on")


class PSQLDatabase:
    pool = None

    @classmethod
    async def get_pool(cls):
        if cls.pool is None:
            pool_args = {"dsn": DSN}
            if POSTGRES_SCHEMA:
                schemas = _parse_schemas(POSTGRES_SCHEMA)
                if schemas:
                    pool_args["server_settings"] = {
                        "search_path": _build_search_path(schemas)
                    }
            cls.pool = await asyncpg.create_pool(**pool_args)
        return cls.pool

    @classmethod
    async def close_pool(cls):
        if cls.pool is not None:
            await cls.pool.close()
            cls.pool = None


async def ensure_vector_indexes():
    """Ensure optional pgvector indexes/migrations when explicitly enabled."""
    table_name = "langchain_pg_embedding"
    column_name = "custom_id"
    index_name = f"idx_{table_name}_{column_name}"
    create_legacy_indexes = _env_flag("PGVECTOR_CREATE_LEGACY_INDEXES")
    migrate_cmetadata_jsonb = _env_flag("PGVECTOR_MIGRATE_CMETADATA_JSONB")
    create_cmetadata_gin_index = _env_flag("PGVECTOR_CREATE_CMETADATA_GIN_INDEX")

    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        if create_legacy_indexes:
            await conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({column_name});
            """
            )

            # Expression index for (cmetadata->>'file_id') - critical for query
            # performance. ExtendedPgVector emits cmetadata->>'file_id' = ...
            # so this B-tree index supports fast equality lookups.
            await conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_file_id
                ON {table_name} ((cmetadata->>'file_id'));
            """
            )
        else:
            logger.info(
                "Skipping legacy vector indexes; set PGVECTOR_CREATE_LEGACY_INDEXES=true to enable"
            )

        if migrate_cmetadata_jsonb:
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
        else:
            logger.info(
                "Skipping cmetadata JSONB migration; set PGVECTOR_MIGRATE_CMETADATA_JSONB=true to enable"
            )

        if create_cmetadata_gin_index:
            cmetadata_type = await conn.fetchval(
                """
                SELECT data_type FROM information_schema.columns
                WHERE table_name = 'langchain_pg_embedding'
                  AND table_schema = current_schema()
                  AND column_name = 'cmetadata';
                """
            )
            if cmetadata_type == "jsonb":
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS ix_cmetadata_gin
                    ON langchain_pg_embedding
                    USING gin (cmetadata jsonb_path_ops);
                    """
                )
            elif cmetadata_type == "json":
                logger.warning(
                    "Skipping cmetadata GIN index because cmetadata uses legacy JSON; "
                    "set PGVECTOR_MIGRATE_CMETADATA_JSONB=true to migrate it first"
                )
            elif cmetadata_type is None:
                logger.warning(
                    "Skipping cmetadata GIN index because langchain_pg_embedding.cmetadata "
                    "was not found in the current schema"
                )
            else:
                logger.warning(
                    "Skipping cmetadata GIN index because cmetadata has unsupported type %s",
                    cmetadata_type,
                )
        else:
            logger.info(
                "Skipping cmetadata GIN index; set PGVECTOR_CREATE_CMETADATA_GIN_INDEX=true to enable"
            )

        logger.info("Vector database startup DDL checks complete")


async def pg_health_check() -> bool:
    try:
        pool = await PSQLDatabase.get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False
