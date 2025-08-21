# tests/dash_assistant/test_ingestion.py
import pytest
import pytest_asyncio
from pathlib import Path
from app.dash_assistant.ingestion.csv_loader import load_dashboards_csv, load_charts_csv
from app.dash_assistant.ingestion.md_loader import load_markdown_dir
from app.dash_assistant.ingestion.enrichment_loader import load_enrichment_yaml

pytestmark = pytest.mark.asyncio


class TestCSVLoader:
    """Test CSV loading functionality."""

    async def test_load_dashboards_csv(self, clean_db, db_connection, fixtures_dir):
        """Test loading dashboards from CSV file."""
        csv_path = fixtures_dir / "dashboards.csv"
        
        # Load dashboards
        dashboards = await load_dashboards_csv(csv_path)
        
        # Verify loaded data structure
        assert len(dashboards) == 5
        
        dashboard = dashboards[0]
        assert dashboard['superset_id'] == '1'
        assert dashboard['dashboard_slug'] == 'user-retention-dashboard'
        assert dashboard['title'] == 'User Retention Dashboard'
        assert 'description' in dashboard
        assert 'owner' in dashboard
        assert 'url' in dashboard

    async def test_load_charts_csv(self, clean_db, db_connection, fixtures_dir):
        """Test loading charts from CSV file."""
        csv_path = fixtures_dir / "charts.csv"
        
        # Load charts
        charts = await load_charts_csv(csv_path)
        
        # Verify loaded data structure
        assert len(charts) == 6
        
        chart = charts[0]
        assert chart['superset_chart_id'] == '101'
        assert chart['parent_dashboard_id'] == '1'
        assert chart['title'] == 'Monthly Retention Rate'
        assert chart['viz_type'] == 'line'
        assert 'sql_text' in chart
        assert 'metrics' in chart
        assert 'dimensions' in chart

    async def test_csv_to_database_integration(self, clean_db, fixtures_dir, mock_index_job):
        """Test full CSV to database integration."""
        dashboards_csv = fixtures_dir / "dashboards.csv"
        charts_csv = fixtures_dir / "charts.csv"
        
        # Use mocked index job
        job = mock_index_job
        await job.load_dashboards_from_csv(dashboards_csv)
        await job.load_charts_from_csv(charts_csv)
        
        # Verify dashboards in database
        dashboard_count = await clean_db.fetchval(
            "SELECT COUNT(*) FROM bi_entity WHERE entity_type = 'dashboard'"
        )
        assert dashboard_count == 5
        
        # Verify charts in database
        chart_count = await clean_db.fetchval("SELECT COUNT(*) FROM bi_chart")
        assert chart_count == 6
        
        # Verify relationships
        dashboard = await clean_db.fetchrow(
            "SELECT * FROM bi_entity WHERE superset_id = '1'"
        )
        assert dashboard is not None
        
        charts_for_dashboard = await clean_db.fetch(
            "SELECT * FROM bi_chart WHERE parent_dashboard_id = $1",
            dashboard['entity_id']
        )
        assert len(charts_for_dashboard) == 2  # Dashboard 1 has 2 charts


class TestMarkdownLoader:
    """Test Markdown loading functionality."""

    async def test_load_markdown_dir(self, clean_db, db_connection, fixtures_dir):
        """Test loading markdown files from directory."""
        md_dir = fixtures_dir / "md"
        
        # Load markdown content
        md_content = await load_markdown_dir(md_dir)
        
        # Verify loaded content
        assert len(md_content) == 3
        assert 'user-retention-dashboard' in md_content
        assert 'revenue-analytics' in md_content
        assert 'product-usage-metrics' in md_content
        
        # Check content structure
        retention_content = md_content['user-retention-dashboard']
        assert 'content' in retention_content
        assert 'User Retention Dashboard' in retention_content['content']
        assert 'cohort behavior' in retention_content['content']

    async def test_markdown_to_database_integration(self, clean_db, db_connection, fixtures_dir):
        """Test markdown content integration with database."""
        # First load dashboards
        dashboards_csv = fixtures_dir / "dashboards.csv"
        job = IndexJob()
        await job.load_dashboards_from_csv(dashboards_csv)
        
        # Then load markdown content
        md_dir = fixtures_dir / "md"
        await job.load_markdown_content(md_dir)
        
        # Verify chunks were created
        chunk_count = await db_connection.fetchval(
            "SELECT COUNT(*) FROM bi_chunk WHERE scope = 'desc'"
        )
        assert chunk_count >= 3  # At least one chunk per markdown file
        
        # Verify content is searchable
        chunks = await db_connection.fetch(
            "SELECT * FROM bi_chunk WHERE content ILIKE '%retention%'"
        )
        assert len(chunks) > 0


class TestEnrichmentLoader:
    """Test YAML enrichment functionality."""

    async def test_load_enrichment_yaml(self, fixtures_dir):
        """Test loading enrichment configuration from YAML."""
        yaml_path = fixtures_dir / "enrichment.yaml"
        
        # Load enrichment config
        enrichment = await load_enrichment_yaml(yaml_path)
        
        # Verify structure
        assert 'dashboards' in enrichment
        assert 'global_rules' in enrichment
        
        # Check specific dashboard enrichment
        retention_enrichment = enrichment['dashboards']['user-retention-dashboard']
        assert retention_enrichment['domain'] == 'growth'
        assert 'retention' in retention_enrichment['tags']
        assert retention_enrichment['priority'] == 'high'

    async def test_enrichment_application(self, clean_db, db_connection, fixtures_dir):
        """Test applying enrichment to database entities."""
        # Load dashboards first
        dashboards_csv = fixtures_dir / "dashboards.csv"
        job = IndexJob()
        await job.load_dashboards_from_csv(dashboards_csv)
        
        # Apply enrichment
        enrichment_yaml = fixtures_dir / "enrichment.yaml"
        await job.apply_enrichment(enrichment_yaml)
        
        # Verify enrichment was applied
        enriched_dashboard = await db_connection.fetchrow(
            "SELECT * FROM bi_entity WHERE dashboard_slug = 'user-retention-dashboard'"
        )
        
        assert enriched_dashboard['domain'] == 'growth'
        assert 'retention' in enriched_dashboard['tags']
        assert 'cohort' in enriched_dashboard['tags']
        
        # Check domain normalization
        revenue_dashboard = await db_connection.fetchrow(
            "SELECT * FROM bi_entity WHERE dashboard_slug = 'revenue-analytics'"
        )
        assert revenue_dashboard['domain'] == 'finance'


class TestChunkCreation:
    """Test chunk creation for different content types."""

    async def test_dashboard_title_chunks(self, clean_db, db_connection, fixtures_dir):
        """Test creation of title chunks for dashboards."""
        dashboards_csv = fixtures_dir / "dashboards.csv"
        job = IndexJob()
        await job.load_dashboards_from_csv(dashboards_csv)
        
        # Verify title chunks were created
        title_chunks = await db_connection.fetch(
            "SELECT * FROM bi_chunk WHERE scope = 'title'"
        )
        assert len(title_chunks) == 5  # One per dashboard
        
        # Check specific title chunk
        retention_chunk = await db_connection.fetchrow(
            """
            SELECT bc.* FROM bi_chunk bc
            JOIN bi_entity be ON bc.entity_id = be.entity_id
            WHERE be.dashboard_slug = 'user-retention-dashboard' AND bc.scope = 'title'
            """
        )
        assert retention_chunk is not None
        assert 'User Retention Dashboard' in retention_chunk['content']

    async def test_chart_title_chunks(self, clean_db, db_connection, fixtures_dir):
        """Test creation of chart title chunks."""
        dashboards_csv = fixtures_dir / "dashboards.csv"
        charts_csv = fixtures_dir / "charts.csv"
        
        job = IndexJob()
        await job.load_dashboards_from_csv(dashboards_csv)
        await job.load_charts_from_csv(charts_csv)
        
        # Verify chart title chunks were created
        chart_title_chunks = await db_connection.fetch(
            "SELECT * FROM bi_chunk WHERE scope = 'chart_title'"
        )
        assert len(chart_title_chunks) == 6  # One per chart
        
        # Check specific chart title chunk
        retention_chart_chunk = await db_connection.fetchrow(
            """
            SELECT bc.* FROM bi_chunk bc
            JOIN bi_chart bch ON bc.chart_id = bch.chart_id
            WHERE bch.superset_chart_id = '101' AND bc.scope = 'chart_title'
            """
        )
        assert retention_chart_chunk is not None
        assert 'Monthly Retention Rate' in retention_chart_chunk['content']

    async def test_sql_chunks(self, clean_db, db_connection, fixtures_dir):
        """Test creation of SQL chunks from chart queries."""
        dashboards_csv = fixtures_dir / "dashboards.csv"
        charts_csv = fixtures_dir / "charts.csv"
        
        job = IndexJob()
        await job.load_dashboards_from_csv(dashboards_csv)
        await job.load_charts_from_csv(charts_csv)
        
        # Verify SQL chunks were created
        sql_chunks = await db_connection.fetch(
            "SELECT * FROM bi_chunk WHERE scope = 'sql'"
        )
        assert len(sql_chunks) == 6  # One per chart (all have SQL)
        
        # Check specific SQL chunk
        sql_chunk = await db_connection.fetchrow(
            """
            SELECT bc.* FROM bi_chunk bc
            JOIN bi_chart bch ON bc.chart_id = bch.chart_id
            WHERE bch.superset_chart_id = '101' AND bc.scope = 'sql'
            """
        )
        assert sql_chunk is not None
        assert 'SELECT DATE_TRUNC' in sql_chunk['content']


class TestUpsertBehavior:
    """Test upsert behavior to prevent duplicates."""

    async def test_dashboard_upsert_by_superset_id(self, clean_db, db_connection, fixtures_dir):
        """Test that dashboards are upserted by superset_id."""
        dashboards_csv = fixtures_dir / "dashboards.csv"
        job = IndexJob()
        
        # Load dashboards twice
        await job.load_dashboards_from_csv(dashboards_csv)
        await job.load_dashboards_from_csv(dashboards_csv)
        
        # Should still have only 5 dashboards
        dashboard_count = await db_connection.fetchval(
            "SELECT COUNT(*) FROM bi_entity WHERE entity_type = 'dashboard'"
        )
        assert dashboard_count == 5

    async def test_chart_upsert_by_superset_chart_id(self, clean_db, db_connection, fixtures_dir):
        """Test that charts are upserted by superset_chart_id."""
        dashboards_csv = fixtures_dir / "dashboards.csv"
        charts_csv = fixtures_dir / "charts.csv"
        
        job = IndexJob()
        await job.load_dashboards_from_csv(dashboards_csv)
        
        # Load charts twice
        await job.load_charts_from_csv(charts_csv)
        await job.load_charts_from_csv(charts_csv)
        
        # Should still have only 6 charts
        chart_count = await db_connection.fetchval("SELECT COUNT(*) FROM bi_chart")
        assert chart_count == 6

    async def test_chunk_recreation_on_upsert(self, clean_db, db_connection, fixtures_dir):
        """Test that chunks are properly recreated on entity upsert."""
        dashboards_csv = fixtures_dir / "dashboards.csv"
        job = IndexJob()
        
        # Load dashboards first time
        await job.load_dashboards_from_csv(dashboards_csv)
        initial_chunk_count = await db_connection.fetchval("SELECT COUNT(*) FROM bi_chunk")
        
        # Load dashboards second time
        await job.load_dashboards_from_csv(dashboards_csv)
        final_chunk_count = await db_connection.fetchval("SELECT COUNT(*) FROM bi_chunk")
        
        # Chunk count should be the same (old chunks replaced)
        assert initial_chunk_count == final_chunk_count
        assert initial_chunk_count > 0


class TestFullIngestionPipeline:
    """Test complete ingestion pipeline."""

    async def test_complete_pipeline_order(self, clean_db, db_connection, fixtures_dir):
        """Test complete ingestion pipeline in correct order."""
        job = IndexJob()
        
        # Run complete pipeline: CSV → MD → enrichment → commit
        await job.run_complete_ingestion(
            dashboards_csv=fixtures_dir / "dashboards.csv",
            charts_csv=fixtures_dir / "charts.csv",
            md_dir=fixtures_dir / "md",
            enrichment_yaml=fixtures_dir / "enrichment.yaml"
        )
        
        # Verify all components are loaded
        dashboard_count = await db_connection.fetchval(
            "SELECT COUNT(*) FROM bi_entity WHERE entity_type = 'dashboard'"
        )
        assert dashboard_count == 5
        
        chart_count = await db_connection.fetchval("SELECT COUNT(*) FROM bi_chart")
        assert chart_count == 6
        
        chunk_count = await db_connection.fetchval("SELECT COUNT(*) FROM bi_chunk")
        assert chunk_count > 10  # Multiple chunk types
        
        # Verify enrichment was applied
        enriched_count = await db_connection.fetchval(
            "SELECT COUNT(*) FROM bi_entity WHERE domain IS NOT NULL"
        )
        assert enriched_count == 5
        
        # Verify different chunk types exist
        chunk_types = await db_connection.fetch(
            "SELECT DISTINCT scope FROM bi_chunk ORDER BY scope"
        )
        scope_names = [row['scope'] for row in chunk_types]
        assert 'title' in scope_names
        assert 'desc' in scope_names
        assert 'chart_title' in scope_names
        assert 'sql' in scope_names

    async def test_pipeline_deterministic_results(self, clean_db, db_connection, fixtures_dir):
        """Test that pipeline produces deterministic results."""
        job = IndexJob()
        
        # Run pipeline twice
        for _ in range(2):
            await job.run_complete_ingestion(
                dashboards_csv=fixtures_dir / "dashboards.csv",
                charts_csv=fixtures_dir / "charts.csv",
                md_dir=fixtures_dir / "md",
                enrichment_yaml=fixtures_dir / "enrichment.yaml"
            )
        
        # Results should be identical
        dashboard_count = await db_connection.fetchval(
            "SELECT COUNT(*) FROM bi_entity WHERE entity_type = 'dashboard'"
        )
        assert dashboard_count == 5
        
        chart_count = await db_connection.fetchval("SELECT COUNT(*) FROM bi_chart")
        assert chart_count == 6
        
        # Check that specific enrichment is consistent
        retention_dashboard = await db_connection.fetchrow(
            "SELECT * FROM bi_entity WHERE dashboard_slug = 'user-retention-dashboard'"
        )
        assert retention_dashboard['domain'] == 'growth'
        assert 'retention' in retention_dashboard['tags']
