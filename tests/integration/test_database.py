import asyncpg
import pytest

from app.services import database


@pytest.mark.integration
async def test_guarded_ddl_uses_schema_and_skips_json_until_migrated(
    monkeypatch, pg_url, caplog
):
    """Guarded DDL targets POSTGRES_SCHEMA and waits for a JSONB column."""
    schema = "test_guarded_startup_ddl"
    dsn = pg_url.replace("postgresql+psycopg2://", "postgresql://")
    admin_conn = await asyncpg.connect(dsn)

    try:
        await admin_conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await admin_conn.execute(f"CREATE SCHEMA {schema}")
        await admin_conn.execute(
            f"""
            CREATE TABLE {schema}.langchain_pg_embedding (
                cmetadata JSON
            )
            """
        )
        monkeypatch.setattr(database, "DSN", dsn)
        monkeypatch.setattr(database, "POSTGRES_SCHEMA", schema)
        monkeypatch.setattr(database.PSQLDatabase, "pool", None)
        monkeypatch.delenv("PGVECTOR_CREATE_LEGACY_INDEXES", raising=False)
        monkeypatch.delenv("PGVECTOR_MIGRATE_CMETADATA_JSONB", raising=False)
        monkeypatch.setenv("PGVECTOR_CREATE_CMETADATA_GIN_INDEX", "true")

        await database.ensure_vector_indexes()

        pool = await database.PSQLDatabase.get_pool()
        async with pool.acquire() as conn:
            assert await conn.fetchval("SELECT current_schema()") == schema
            index_exists = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1 FROM pg_indexes
                    WHERE schemaname = current_schema()
                      AND indexname = 'ix_cmetadata_gin'
                )
                """
            )
        assert index_exists is False
        assert "uses legacy JSON" in caplog.text
        assert "PGVECTOR_MIGRATE_CMETADATA_JSONB=true" in caplog.text

        monkeypatch.setenv("PGVECTOR_MIGRATE_CMETADATA_JSONB", "true")
        await database.ensure_vector_indexes()

        async with pool.acquire() as conn:
            column_type = await conn.fetchval(
                """
                SELECT data_type FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = 'langchain_pg_embedding'
                  AND column_name = 'cmetadata'
                """
            )
            index_exists = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1 FROM pg_indexes
                    WHERE schemaname = current_schema()
                      AND indexname = 'ix_cmetadata_gin'
                )
                """
            )
        assert column_type == "jsonb"
        assert index_exists is True
    finally:
        await database.PSQLDatabase.close_pool()
        await admin_conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await admin_conn.close()
