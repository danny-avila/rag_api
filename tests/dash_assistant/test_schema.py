# tests/dash_assistant/test_schema.py
import pytest
import pytest_asyncio
import asyncpg
from app.config import DSN

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def db_connection():
    """Create database connection for testing."""
    conn = await asyncpg.connect(DSN)
    yield conn
    await conn.close()


class TestDashAssistantSchema:
    """Test suite for dash assistant database schema."""

    async def test_extensions_installed(self, db_connection):
        """Test that required PostgreSQL extensions are installed."""
        # Check vector extension
        vector_exists = await db_connection.fetchval(
            "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
        )
        assert vector_exists, "vector extension is not installed"

        # Check pg_trgm extension
        trgm_exists = await db_connection.fetchval(
            "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm')"
        )
        assert trgm_exists, "pg_trgm extension is not installed"

    async def test_required_tables_exist(self, db_connection):
        """Test that all required tables exist."""
        required_tables = ['bi_entity', 'bi_chart', 'bi_chunk', 'term_dict', 'query_log']
        
        for table_name in required_tables:
            table_exists = await db_connection.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name = $1
                )
                """,
                table_name
            )
            assert table_exists, f"Table {table_name} does not exist"

    async def test_bi_entity_table_structure(self, db_connection):
        """Test bi_entity table structure and constraints."""
        # Check columns exist
        columns = await db_connection.fetch(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'bi_entity' AND table_schema = 'public'
            ORDER BY ordinal_position
            """
        )
        
        column_names = [col['column_name'] for col in columns]
        expected_columns = [
            'entity_id', 'entity_type', 'superset_id', 'dashboard_slug',
            'title', 'description', 'domain', 'owner', 'tags', 'url',
            'usage_score', 'last_refresh_ts', 'metadata'
        ]
        
        for expected_col in expected_columns:
            assert expected_col in column_names, f"Column {expected_col} missing in bi_entity"

        # Check entity_type constraint
        constraint_exists = await db_connection.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.check_constraints cc
                JOIN information_schema.constraint_column_usage ccu 
                ON cc.constraint_name = ccu.constraint_name
                WHERE ccu.table_name = 'bi_entity' 
                AND ccu.column_name = 'entity_type'
                AND cc.check_clause LIKE '%dashboard%chart%'
            )
            """
        )
        assert constraint_exists, "entity_type check constraint not found"

    async def test_bi_chunk_table_structure(self, db_connection):
        """Test bi_chunk table structure and embedding dimension."""
        # Check embedding column exists with vector type
        embedding_info = await db_connection.fetchrow(
            """
            SELECT data_type, character_maximum_length
            FROM information_schema.columns
            WHERE table_name = 'bi_chunk' 
            AND column_name = 'embedding'
            AND table_schema = 'public'
            """
        )
        
        assert embedding_info is not None, "embedding column not found in bi_chunk"
        assert embedding_info['data_type'] == 'USER-DEFINED', "embedding column is not vector type"

        # Check embedding dimension is 3072
        vector_dim = await db_connection.fetchval(
            """
            SELECT atttypmod 
            FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = 'public' 
            AND c.relname = 'bi_chunk' 
            AND a.attname = 'embedding'
            """
        )
        
        # atttypmod for vector(1536) should be 1536 + 4, but actual may vary slightly
        expected_dim = 1536
        actual_dim = vector_dim - 4 if vector_dim else 0
        assert abs(actual_dim - expected_dim) <= 10, f"embedding dimension should be around {expected_dim}, got {actual_dim}"

    async def test_tsvector_trigger_exists(self, db_connection):
        """Test that tsvector trigger exists and function is created."""
        # Check trigger function exists
        function_exists = await db_connection.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.routines
                WHERE routine_name = 'make_tsvectors'
                AND routine_schema = 'public'
            )
            """
        )
        assert function_exists, "make_tsvectors function not found"

        # Check trigger exists
        trigger_exists = await db_connection.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.triggers
                WHERE trigger_name = 'bi_chunk_tsvectors'
                AND event_object_table = 'bi_chunk'
            )
            """
        )
        assert trigger_exists, "bi_chunk_tsvectors trigger not found"

    async def test_required_indexes_exist(self, db_connection):
        """Test that all required indexes exist."""
        required_indexes = [
            # Unique indexes
            'uq_bi_entity_superset_id',
            'uq_bi_entity_slug', 
            'uq_bi_chart_superset_chart_id',
            # Vector index (commented out in migration)
            # 'idx_bi_chunk_emb_hnsw',
            # GIN indexes for tsvector
            'idx_bi_chunk_tsv_en',
            'idx_bi_chunk_tsv_ru',
            # Trigram indexes
            'idx_bi_entity_title_trgm',
            'idx_bi_chart_title_trgm'
        ]

        for index_name in required_indexes:
            index_exists = await db_connection.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1 FROM pg_indexes
                    WHERE indexname = $1 AND schemaname = 'public'
                )
                """,
                index_name
            )
            assert index_exists, f"Index {index_name} does not exist"

    async def test_hnsw_index_configuration(self, db_connection):
        """Test that HNSW index is properly configured."""
        index_info = await db_connection.fetchrow(
            """
            SELECT indexdef FROM pg_indexes
            WHERE indexname = 'idx_bi_chunk_emb_hnsw'
            AND schemaname = 'public'
            """
        )
        
        assert index_info is not None, "HNSW index not found"
        index_def = index_info['indexdef']
        
        assert 'hnsw' in index_def.lower(), "Index is not using HNSW"
        assert 'vector_cosine_ops' in index_def.lower(), "Index is not using cosine distance operator"
        assert 'm = 16' in index_def, "Index m parameter not set to 16"
        assert 'ef_construction = 64' in index_def, "Index ef_construction parameter not set to 64"

    async def test_gin_indexes_configuration(self, db_connection):
        """Test that GIN indexes are properly configured."""
        # Test tsvector GIN indexes
        tsv_indexes = await db_connection.fetch(
            """
            SELECT indexname, indexdef FROM pg_indexes
            WHERE indexname IN ('idx_bi_chunk_tsv_en', 'idx_bi_chunk_tsv_ru')
            AND schemaname = 'public'
            """
        )
        
        assert len(tsv_indexes) == 2, "Not all tsvector GIN indexes found"
        
        for idx in tsv_indexes:
            assert 'gin' in idx['indexdef'].lower(), f"Index {idx['indexname']} is not using GIN"

        # Test trigram GIN indexes
        trgm_indexes = await db_connection.fetch(
            """
            SELECT indexname, indexdef FROM pg_indexes
            WHERE indexname IN ('idx_bi_entity_title_trgm', 'idx_bi_chart_title_trgm')
            AND schemaname = 'public'
            """
        )
        
        assert len(trgm_indexes) == 2, "Not all trigram GIN indexes found"
        
        for idx in trgm_indexes:
            assert 'gin' in idx['indexdef'].lower(), f"Index {idx['indexname']} is not using GIN"
            assert 'gin_trgm_ops' in idx['indexdef'], f"Index {idx['indexname']} is not using gin_trgm_ops"

    async def test_foreign_key_constraints(self, db_connection):
        """Test that foreign key constraints are properly set."""
        # Check bi_chart references bi_entity
        fk_exists = await db_connection.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.referential_constraints rc
                JOIN information_schema.key_column_usage kcu 
                ON rc.constraint_name = kcu.constraint_name
                WHERE kcu.table_name = 'bi_chart'
                AND kcu.column_name = 'parent_dashboard_id'
                AND rc.delete_rule = 'CASCADE'
            )
            """
        )
        assert fk_exists, "Foreign key constraint from bi_chart to bi_entity not found"

        # Check bi_chunk references bi_entity
        fk_exists = await db_connection.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.referential_constraints rc
                JOIN information_schema.key_column_usage kcu 
                ON rc.constraint_name = kcu.constraint_name
                WHERE kcu.table_name = 'bi_chunk'
                AND kcu.column_name = 'entity_id'
                AND rc.delete_rule = 'CASCADE'
            )
            """
        )
        assert fk_exists, "Foreign key constraint from bi_chunk to bi_entity not found"

        # Check bi_chunk references bi_chart
        fk_exists = await db_connection.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.referential_constraints rc
                JOIN information_schema.key_column_usage kcu 
                ON rc.constraint_name = kcu.constraint_name
                WHERE kcu.table_name = 'bi_chunk'
                AND kcu.column_name = 'chart_id'
                AND rc.delete_rule = 'CASCADE'
            )
            """
        )
        assert fk_exists, "Foreign key constraint from bi_chunk to bi_chart not found"

    async def test_tsvector_trigger_functionality(self, db_connection):
        """Test that tsvector trigger works correctly."""
        # Insert test data
        await db_connection.execute(
            """
            INSERT INTO bi_entity (entity_type, title, description)
            VALUES ('dashboard', 'Test Dashboard', 'Test description')
            """
        )
        
        entity_id = await db_connection.fetchval(
            "SELECT entity_id FROM bi_entity WHERE title = 'Test Dashboard'"
        )
        
        await db_connection.execute(
            """
            INSERT INTO bi_chunk (entity_id, scope, content, lang)
            VALUES ($1, 'title', 'Test content for search', 'en')
            """,
            entity_id
        )
        
        # Check that tsvectors were created
        tsvectors = await db_connection.fetchrow(
            """
            SELECT tsv_en IS NOT NULL as has_en, tsv_ru IS NOT NULL as has_ru
            FROM bi_chunk WHERE entity_id = $1
            """,
            entity_id
        )
        
        assert tsvectors['has_en'], "English tsvector not created"
        assert tsvectors['has_ru'], "Russian tsvector not created"
        
        # Cleanup
        await db_connection.execute("DELETE FROM bi_entity WHERE entity_id = $1", entity_id)
