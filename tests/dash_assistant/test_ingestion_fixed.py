# tests/dash_assistant/test_ingestion_fixed.py
"""Fixed ingestion tests with proper async handling."""
import pytest
import pytest_asyncio
from pathlib import Path
from app.dash_assistant.ingestion.csv_loader import load_dashboards_csv, load_charts_csv
from app.dash_assistant.ingestion.md_loader import load_markdown_dir
from app.dash_assistant.ingestion.enrichment_loader import load_enrichment_yaml

pytestmark = pytest.mark.asyncio


class TestCSVLoaderFixed:
    """Test CSV loading functionality with proper isolation."""

    async def test_load_dashboards_csv(self, fixtures_dir):
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

    async def test_load_charts_csv(self, fixtures_dir):
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

    async def test_database_integration_isolated(self, clean_db, fixtures_dir, mock_index_job):
        """Test database integration with proper isolation."""
        dashboards_csv = fixtures_dir / "dashboards.csv"
        charts_csv = fixtures_dir / "charts.csv"
        
        # Use mocked index job
        job = mock_index_job
        
        # Load dashboards
        dashboard_count = await job.load_dashboards_from_csv(dashboards_csv)
        assert dashboard_count == 5
        
        # Verify in database
        db_dashboard_count = await clean_db.fetchval(
            "SELECT COUNT(*) FROM bi_entity WHERE entity_type = 'dashboard'"
        )
        assert db_dashboard_count == 5
        
        # Load charts
        chart_count = await job.load_charts_from_csv(charts_csv)
        assert chart_count == 6
        
        # Verify in database
        db_chart_count = await clean_db.fetchval("SELECT COUNT(*) FROM bi_chart")
        assert db_chart_count == 6


class TestMarkdownLoaderFixed:
    """Test Markdown loading functionality."""

    async def test_load_markdown_dir(self, fixtures_dir):
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

    async def test_markdown_integration_isolated(self, clean_db, fixtures_dir, mock_index_job):
        """Test markdown integration with proper isolation."""
        # First load dashboards
        dashboards_csv = fixtures_dir / "dashboards.csv"
        job = mock_index_job
        await job.load_dashboards_from_csv(dashboards_csv)
        
        # Then load markdown content
        md_dir = fixtures_dir / "md"
        processed_count = await job.load_markdown_content(md_dir)
        assert processed_count == 3
        
        # Verify chunks were created
        chunk_count = await clean_db.fetchval(
            "SELECT COUNT(*) FROM bi_chunk WHERE scope = 'desc'"
        )
        assert chunk_count >= 3  # At least one chunk per markdown file


class TestEnrichmentLoaderFixed:
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

    async def test_enrichment_application_isolated(self, clean_db, fixtures_dir, mock_index_job):
        """Test applying enrichment with proper isolation."""
        # Load dashboards first
        dashboards_csv = fixtures_dir / "dashboards.csv"
        job = mock_index_job
        await job.load_dashboards_from_csv(dashboards_csv)
        
        # Apply enrichment
        enrichment_yaml = fixtures_dir / "enrichment.yaml"
        enriched_count = await job.apply_enrichment(enrichment_yaml)
        assert enriched_count == 5
        
        # Verify enrichment was applied
        enriched_dashboard = await clean_db.fetchrow(
            "SELECT * FROM bi_entity WHERE dashboard_slug = 'user-retention-dashboard'"
        )
        
        assert enriched_dashboard['domain'] == 'growth'
        assert 'retention' in enriched_dashboard['tags']
        assert 'cohort' in enriched_dashboard['tags']


class TestUpsertBehaviorFixed:
    """Test upsert behavior with proper isolation."""

    async def test_dashboard_upsert_isolated(self, clean_db, fixtures_dir, mock_index_job):
        """Test that dashboards are upserted by superset_id."""
        dashboards_csv = fixtures_dir / "dashboards.csv"
        job = mock_index_job
        
        # Load dashboards twice
        await job.load_dashboards_from_csv(dashboards_csv)
        await job.load_dashboards_from_csv(dashboards_csv)
        
        # Should still have only 5 dashboards
        dashboard_count = await clean_db.fetchval(
            "SELECT COUNT(*) FROM bi_entity WHERE entity_type = 'dashboard'"
        )
        assert dashboard_count == 5

    async def test_chart_upsert_isolated(self, clean_db, fixtures_dir, mock_index_job):
        """Test that charts are upserted by superset_chart_id."""
        dashboards_csv = fixtures_dir / "dashboards.csv"
        charts_csv = fixtures_dir / "charts.csv"
        
        job = mock_index_job
        await job.load_dashboards_from_csv(dashboards_csv)
        
        # Load charts twice
        await job.load_charts_from_csv(charts_csv)
        await job.load_charts_from_csv(charts_csv)
        
        # Should still have only 6 charts
        chart_count = await clean_db.fetchval("SELECT COUNT(*) FROM bi_chart")
        assert chart_count == 6


class TestFullPipelineFixed:
    """Test complete ingestion pipeline with proper isolation."""

    async def test_complete_pipeline_isolated(self, clean_db, fixtures_dir, mock_index_job):
        """Test complete ingestion pipeline."""
        job = mock_index_job
        
        # Run complete pipeline step by step
        results = await job.run_complete_ingestion(
            dashboards_csv=fixtures_dir / "dashboards.csv",
            charts_csv=fixtures_dir / "charts.csv",
            md_dir=fixtures_dir / "md",
            enrichment_yaml=fixtures_dir / "enrichment.yaml"
        )
        
        # Verify results
        assert results['dashboards_loaded'] == 5
        assert results['charts_loaded'] == 6
        assert results['markdown_processed'] == 3
        assert results['dashboards_enriched'] == 5
        
        # Verify all components are loaded
        dashboard_count = await clean_db.fetchval(
            "SELECT COUNT(*) FROM bi_entity WHERE entity_type = 'dashboard'"
        )
        assert dashboard_count == 5
        
        chart_count = await clean_db.fetchval("SELECT COUNT(*) FROM bi_chart")
        assert chart_count == 6
        
        chunk_count = await clean_db.fetchval("SELECT COUNT(*) FROM bi_chunk")
        assert chunk_count > 10  # Multiple chunk types
        
        # Verify enrichment was applied
        enriched_count = await clean_db.fetchval(
            "SELECT COUNT(*) FROM bi_entity WHERE domain IS NOT NULL"
        )
        assert enriched_count == 5

    async def test_pipeline_deterministic_isolated(self, clean_db, fixtures_dir, mock_index_job):
        """Test that pipeline produces deterministic results."""
        job = mock_index_job
        
        # Run pipeline twice
        for _ in range(2):
            await job.run_complete_ingestion(
                dashboards_csv=fixtures_dir / "dashboards.csv",
                charts_csv=fixtures_dir / "charts.csv",
                md_dir=fixtures_dir / "md",
                enrichment_yaml=fixtures_dir / "enrichment.yaml"
            )
        
        # Results should be identical
        dashboard_count = await clean_db.fetchval(
            "SELECT COUNT(*) FROM bi_entity WHERE entity_type = 'dashboard'"
        )
        assert dashboard_count == 5
        
        chart_count = await clean_db.fetchval("SELECT COUNT(*) FROM bi_chart")
        assert chart_count == 6
        
        # Check that specific enrichment is consistent
        retention_dashboard = await clean_db.fetchrow(
            "SELECT * FROM bi_entity WHERE dashboard_slug = 'user-retention-dashboard'"
        )
        assert retention_dashboard['domain'] == 'growth'
        assert 'retention' in retention_dashboard['tags']
