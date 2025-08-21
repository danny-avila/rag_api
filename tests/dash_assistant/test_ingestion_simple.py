# tests/dash_assistant/test_ingestion_simple.py
"""Simplified ingestion tests using existing patterns."""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from app.dash_assistant.ingestion.csv_loader import load_dashboards_csv, load_charts_csv
from app.dash_assistant.ingestion.md_loader import load_markdown_dir
from app.dash_assistant.ingestion.enrichment_loader import load_enrichment_yaml
from app.dash_assistant.ingestion.index_jobs import IndexJob

pytestmark = pytest.mark.asyncio


@pytest.fixture
def fixtures_dir():
    """Get path to test fixtures directory."""
    return Path(__file__).parent.parent / "fixtures" / "superset"


class TestCSVLoaderSimple:
    """Test CSV loading functionality without database."""

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


class TestMarkdownLoaderSimple:
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


class TestEnrichmentLoaderSimple:
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


class TestIndexJobMocked:
    """Test IndexJob with mocked database operations."""

    @pytest.fixture
    def mock_db(self):
        """Mock database operations."""
        mock_db = AsyncMock()
        
        # Mock database responses
        mock_db.fetch_one.return_value = None  # No existing records
        mock_db.fetch_value.return_value = 1   # Return entity_id
        mock_db.fetch_all.return_value = []    # Empty results
        mock_db.execute_query.return_value = None
        
        return mock_db

    async def test_dashboard_loading_logic(self, fixtures_dir, mock_db):
        """Test dashboard loading logic with mocked database."""
        with patch('app.dash_assistant.ingestion.index_jobs.DashAssistantDB', mock_db):
            job = IndexJob()
            dashboards_csv = fixtures_dir / "dashboards.csv"
            
            # This should work without database connection issues
            count = await job.load_dashboards_from_csv(dashboards_csv)
            
            # Verify the logic worked
            assert count == 5
            
            # Verify database operations were called
            assert mock_db.fetch_one.called
            assert mock_db.fetch_value.called
            assert mock_db.execute_query.called

    async def test_chart_loading_logic(self, fixtures_dir, mock_db):
        """Test chart loading logic with mocked database."""
        # Mock parent dashboard exists and chart creation
        mock_db.fetch_one.return_value = {'entity_id': 1}
        mock_db.fetch_value.return_value = 1  # chart_id
        
        with patch('app.dash_assistant.ingestion.index_jobs.DashAssistantDB', mock_db):
            job = IndexJob()
            charts_csv = fixtures_dir / "charts.csv"
            
            count = await job.load_charts_from_csv(charts_csv)
            
            # Verify the logic worked
            assert count == 6
            
            # Verify database operations were called
            assert mock_db.fetch_one.called
            assert mock_db.fetch_value.called
            assert mock_db.execute_query.called

    async def test_markdown_processing_logic(self, fixtures_dir, mock_db):
        """Test markdown processing logic with mocked database."""
        # Mock entity exists
        mock_db.fetch_one.return_value = {'entity_id': 1}
        
        with patch('app.dash_assistant.ingestion.index_jobs.DashAssistantDB', mock_db):
            job = IndexJob()
            md_dir = fixtures_dir / "md"
            
            count = await job.load_markdown_content(md_dir)
            
            # Verify the logic worked
            assert count == 3
            
            # Verify database operations were called
            assert mock_db.fetch_one.called
            assert mock_db.execute_query.called

    async def test_enrichment_logic(self, fixtures_dir, mock_db):
        """Test enrichment logic with mocked database."""
        # Mock existing dashboards
        mock_dashboards = [
            {'entity_id': 1, 'dashboard_slug': 'user-retention-dashboard', 'tags': None},
            {'entity_id': 2, 'dashboard_slug': 'revenue-analytics', 'tags': None},
            {'entity_id': 3, 'dashboard_slug': 'product-usage-metrics', 'tags': None},
            {'entity_id': 4, 'dashboard_slug': 'marketing-performance', 'tags': None},
            {'entity_id': 5, 'dashboard_slug': 'operational-kpis', 'tags': None},
        ]
        mock_db.fetch_all.return_value = mock_dashboards
        
        with patch('app.dash_assistant.ingestion.index_jobs.DashAssistantDB', mock_db):
            job = IndexJob()
            enrichment_yaml = fixtures_dir / "enrichment.yaml"
            
            count = await job.apply_enrichment(enrichment_yaml)
            
            # Verify the logic worked
            assert count == 5
            
            # Verify database operations were called
            assert mock_db.fetch_all.called
            assert mock_db.execute_query.called

    async def test_complete_pipeline_logic(self, fixtures_dir, mock_db):
        """Test complete pipeline logic with mocked database."""
        # Setup mocks for different operations
        mock_db.fetch_one.side_effect = [
            None,  # No existing dashboard
            {'entity_id': 1},  # Parent dashboard exists for charts
            {'entity_id': 1},  # Entity exists for markdown
        ]
        mock_db.fetch_value.return_value = 1
        mock_db.fetch_all.return_value = [
            {'entity_id': 1, 'dashboard_slug': 'user-retention-dashboard', 'tags': None}
        ]
        
        with patch('app.dash_assistant.ingestion.index_jobs.DashAssistantDB', mock_db):
            job = IndexJob()
            
            results = await job.run_complete_ingestion(
                dashboards_csv=fixtures_dir / "dashboards.csv",
                charts_csv=fixtures_dir / "charts.csv",
                md_dir=fixtures_dir / "md",
                enrichment_yaml=fixtures_dir / "enrichment.yaml"
            )
            
            # Verify results structure
            assert 'dashboards_loaded' in results
            assert 'charts_loaded' in results
            assert 'markdown_processed' in results
            assert 'dashboards_enriched' in results
            
            # Verify all operations were attempted
            assert mock_db.fetch_one.called
            assert mock_db.fetch_value.called
            assert mock_db.fetch_all.called
            assert mock_db.execute_query.called


@pytest.mark.skip_asyncio
class TestValidationLogic:
    """Test validation and normalization logic."""

    def test_dashboard_validation(self):
        """Test dashboard data validation."""
        from app.dash_assistant.ingestion.validators import validate_dashboard_data
        
        # Valid dashboard
        valid_dashboard = {
            'title': 'Test Dashboard',
            'superset_id': '123',
            'url': 'https://example.com/dashboard/123'
        }
        errors = validate_dashboard_data(valid_dashboard)
        assert len(errors) == 0
        
        # Invalid dashboard
        invalid_dashboard = {
            'title': '',  # Empty title
            'superset_id': 'abc',  # Non-numeric
            'url': 'invalid-url'  # Invalid URL
        }
        errors = validate_dashboard_data(invalid_dashboard)
        assert len(errors) > 0
        assert 'title' in errors
        assert 'superset_id' in errors
        assert 'url' in errors

    def test_chart_validation(self):
        """Test chart data validation."""
        from app.dash_assistant.ingestion.validators import validate_chart_data
        
        # Valid chart
        valid_chart = {
            'title': 'Test Chart',
            'superset_chart_id': '456',
            'parent_dashboard_id': '123'
        }
        errors = validate_chart_data(valid_chart)
        assert len(errors) == 0
        
        # Invalid chart
        invalid_chart = {
            'title': '',  # Empty title
            'superset_chart_id': '',  # Empty ID
            'parent_dashboard_id': ''  # Empty parent ID
        }
        errors = validate_chart_data(invalid_chart)
        assert len(errors) > 0

    def test_normalization_functions(self):
        """Test data normalization functions."""
        from app.dash_assistant.ingestion.validators import (
            normalize_slug, normalize_domain, normalize_tags
        )
        
        # Test slug normalization
        assert normalize_slug("User Retention Dashboard") == "user-retention-dashboard"
        assert normalize_slug("Revenue_Analytics") == "revenue-analytics"
        
        # Test domain normalization
        domain_mapping = {"fin": "finance", "ops": "operations"}
        assert normalize_domain("fin", domain_mapping) == "finance"
        assert normalize_domain("product", domain_mapping) == "product"
        
        # Test tags normalization
        tags = ["Retention", "COHORT", "user-analytics"]
        normalized = normalize_tags(tags)
        assert "retention" in normalized
        assert "cohort" in normalized
        assert "user-analytics" in normalized
        assert len(normalized) == len(set(normalized))  # No duplicates


class TestUpsertLogic:
    """Test upsert behavior logic."""
    
    @pytest.fixture
    def mock_db(self):
        """Mock database operations."""
        mock_db = AsyncMock()
        
        # Mock database responses
        mock_db.fetch_one.return_value = None  # No existing records
        mock_db.fetch_value.return_value = 1   # Return entity_id
        mock_db.fetch_all.return_value = []    # Empty results
        mock_db.execute_query.return_value = None
        
        return mock_db

    async def test_upsert_logic_new_record(self, mock_db):
        """Test upsert logic for new records."""
        # Mock no existing record
        mock_db.fetch_one.return_value = None
        mock_db.fetch_value.return_value = 1
        
        with patch('app.dash_assistant.ingestion.index_jobs.DashAssistantDB', mock_db):
            job = IndexJob()
            
            dashboard_data = {
                'superset_id': '123',
                'title': 'Test Dashboard',
                'description': 'Test description'
            }
            
            entity_id = await job._upsert_dashboard(dashboard_data)
            assert entity_id == 1
            
            # Should call INSERT, not UPDATE
            insert_calls = [call for call in mock_db.execute_query.call_args_list 
                          if 'INSERT' in str(call)]
            assert len(insert_calls) > 0

    async def test_upsert_logic_existing_record(self, mock_db):
        """Test upsert logic for existing records."""
        # Mock existing record
        mock_db.fetch_one.return_value = {'entity_id': 1}
        
        with patch('app.dash_assistant.ingestion.index_jobs.DashAssistantDB', mock_db):
            job = IndexJob()
            
            dashboard_data = {
                'superset_id': '123',
                'title': 'Updated Dashboard',
                'description': 'Updated description'
            }
            
            entity_id = await job._upsert_dashboard(dashboard_data)
            assert entity_id == 1
            
            # Should call UPDATE, not INSERT
            update_calls = [call for call in mock_db.execute_query.call_args_list 
                          if 'UPDATE' in str(call)]
            assert len(update_calls) > 0
