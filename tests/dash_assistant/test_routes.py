# tests/dash_assistant/test_routes.py
"""Tests for dash assistant FastAPI routes."""
import pytest
import json
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock

# Import the main app
from main import app


class TestDashAssistantRoutes:
    """Test dash assistant API routes."""
    
    @pytest.fixture
    def client(self):
        """Create FastAPI test client."""
        return TestClient(app)
    
    @pytest.fixture
    def fixtures_path(self):
        """Path to test fixtures."""
        return Path(__file__).parent.parent / "fixtures" / "superset"
    
    def test_ingest_endpoint_success(self, client, fixtures_path):
        """Test successful ingestion with fixture paths."""
        # Mock the ingestion job
        with patch('app.dash_assistant.serving.routes.IndexJob') as mock_job_class:
            mock_job = AsyncMock()
            mock_job_class.return_value = mock_job
            
            # Mock the ingestion methods
            mock_job.load_dashboards_from_csv.return_value = 3
            mock_job.load_charts_from_csv.return_value = 5
            mock_job.load_markdown_content.return_value = 3
            mock_job.apply_enrichment.return_value = 3
            mock_job.index_missing_chunks.return_value = 10
            
            # Mock database health check
            with patch('app.dash_assistant.serving.routes.DashAssistantDB.health_check', return_value=True):
                payload = {
                    "dashboards_csv": str(fixtures_path / "dashboards.csv"),
                    "charts_csv": str(fixtures_path / "charts.csv"),
                    "md_dir": str(fixtures_path / "md"),
                    "enrichment_yaml": str(fixtures_path / "enrichment.yaml"),
                    "run_embeddings": True
                }
                
                response = client.post("/dash/ingest", json=payload)
                
                assert response.status_code == 200
                data = response.json()
                
                assert "status" in data
                assert data["status"] == "success"
                assert "results" in data
                assert data["results"]["dashboards_loaded"] == 3
                assert data["results"]["charts_loaded"] == 5
                assert data["results"]["markdown_processed"] == 3
                assert data["results"]["dashboards_enriched"] == 3
                assert data["results"]["chunks_indexed"] == 10
    
    def test_ingest_endpoint_missing_files(self, client):
        """Test ingestion with missing files."""
        payload = {
            "dashboards_csv": "/nonexistent/dashboards.csv",
            "charts_csv": "/nonexistent/charts.csv",
            "md_dir": "/nonexistent/md",
            "enrichment_yaml": "/nonexistent/enrichment.yaml"
        }
        
        response = client.post("/dash/ingest", json=payload)
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "does not exist" in data["error"]
    
    def test_ingest_endpoint_without_embeddings(self, client, fixtures_path):
        """Test ingestion without running embeddings."""
        with patch('app.dash_assistant.serving.routes.IndexJob') as mock_job_class:
            mock_job = AsyncMock()
            mock_job_class.return_value = mock_job
            
            mock_job.load_dashboards_from_csv.return_value = 3
            mock_job.load_charts_from_csv.return_value = 5
            mock_job.load_markdown_content.return_value = 3
            mock_job.apply_enrichment.return_value = 3
            
            with patch('app.dash_assistant.serving.routes.DashAssistantDB.health_check', return_value=True):
                payload = {
                    "dashboards_csv": str(fixtures_path / "dashboards.csv"),
                    "charts_csv": str(fixtures_path / "charts.csv"),
                    "md_dir": str(fixtures_path / "md"),
                    "enrichment_yaml": str(fixtures_path / "enrichment.yaml"),
                    "run_embeddings": False
                }
                
                response = client.post("/dash/ingest", json=payload)
                
                assert response.status_code == 200
                data = response.json()
                
                assert data["status"] == "success"
                assert "chunks_indexed" not in data["results"]
                
                # Verify embeddings indexer was not called
                mock_job.index_missing_chunks.assert_not_called()
    
    def test_query_endpoint_success(self, client):
        """Test successful query with mocked retriever."""
        # Mock the retriever and answer builder
        with patch('app.dash_assistant.serving.routes.DashRetriever') as mock_retriever_class:
            mock_retriever = AsyncMock()
            mock_retriever_class.return_value = mock_retriever
            
            # Mock search results
            mock_candidates = [
                {
                    'entity_id': 1,
                    'title': 'User Retention Dashboard',
                    'url': 'https://superset.example.com/dashboard/1',
                    'score': 0.95,
                    'usage_score': 10.5,
                    'signal_sources': ['fts', 'vector'],
                    'matched_tokens': ['retention', 'user'],
                    'charts': [
                        {
                            'chart_id': 101,
                            'title': 'Retention Cohort Analysis',
                            'url': 'https://superset.example.com/chart/101',
                            'filters_default': {'time_range': 'last_90_days'}
                        }
                    ]
                }
            ]
            
            mock_retriever.search.return_value = mock_candidates
            
            with patch('app.dash_assistant.serving.routes.DashAssistantDB.health_check', return_value=True):
                payload = {
                    "q": "где найти retention",
                    "top_k": 3
                }
                
                response = client.post("/dash/query", json=payload)
                
                assert response.status_code == 200
                data = response.json()
                
                # Check response structure
                assert "results" in data
                assert "debug" in data
                assert isinstance(data["results"], list)
                assert len(data["results"]) == 1
                
                # Check result structure
                result = data["results"][0]
                assert "title" in result
                assert "url" in result
                assert "score" in result
                assert "charts" in result
                assert "why" in result
                
                assert result["title"] == "User Retention Dashboard"
                assert result["score"] == 0.95
                assert len(result["charts"]) == 1
                assert "retention" in result["why"]
                
                # Check debug info
                assert "query" in data["debug"]
                assert "top_k" in data["debug"]
                assert "candidates_found" in data["debug"]
                assert data["debug"]["query"] == "где найти retention"
                assert data["debug"]["top_k"] == 3
                assert data["debug"]["candidates_found"] == 1
    
    def test_query_endpoint_no_results(self, client):
        """Test query with no results found."""
        with patch('app.dash_assistant.serving.routes.DashRetriever') as mock_retriever_class:
            mock_retriever = AsyncMock()
            mock_retriever_class.return_value = mock_retriever
            
            # Mock empty search results
            mock_retriever.search.return_value = []
            
            with patch('app.dash_assistant.serving.routes.DashAssistantDB.health_check', return_value=True):
                payload = {
                    "q": "nonexistent dashboard",
                    "top_k": 3
                }
                
                response = client.post("/dash/query", json=payload)
                
                assert response.status_code == 200
                data = response.json()
                
                assert "results" in data
                assert "debug" in data
                assert data["results"] == []
                assert data["debug"]["candidates_found"] == 0
    
    def test_query_endpoint_validation_error(self, client):
        """Test query endpoint with invalid payload."""
        # Missing required 'q' field
        payload = {
            "top_k": 3
        }
        
        response = client.post("/dash/query", json=payload)
        
        assert response.status_code == 422
    
    def test_query_endpoint_with_filters(self, client):
        """Test query endpoint with additional filters."""
        with patch('app.dash_assistant.serving.routes.DashRetriever') as mock_retriever_class:
            mock_retriever = AsyncMock()
            mock_retriever_class.return_value = mock_retriever
            
            mock_candidates = [
                {
                    'entity_id': 2,
                    'title': 'Sales Analytics Dashboard',
                    'url': 'https://superset.example.com/dashboard/2',
                    'score': 0.88,
                    'usage_score': 5.0,
                    'signal_sources': ['vector'],
                    'matched_tokens': [],
                    'charts': []
                }
            ]
            
            mock_retriever.search.return_value = mock_candidates
            
            with patch('app.dash_assistant.serving.routes.DashAssistantDB.health_check', return_value=True):
                payload = {
                    "q": "sales dashboard",
                    "top_k": 5,
                    "filters": {
                        "domain": "finance",
                        "owner": "john.doe"
                    }
                }
                
                response = client.post("/dash/query", json=payload)
                
                assert response.status_code == 200
                data = response.json()
                
                assert len(data["results"]) == 1
                assert data["debug"]["top_k"] == 5
                
                # Verify retriever was called with filters
                mock_retriever.search.assert_called_once()
                call_args = mock_retriever.search.call_args
                assert call_args[1]["filters"] == {"domain": "finance", "owner": "john.doe"}
    
    def test_database_health_check_failure(self, client):
        """Test endpoints when database health check fails."""
        with patch('app.dash_assistant.serving.routes.DashAssistantDB.health_check', return_value=False):
            # Test ingest endpoint
            payload = {
                "dashboards_csv": "/some/path.csv",
                "charts_csv": "/some/path.csv",
                "md_dir": "/some/path",
                "enrichment_yaml": "/some/path.yaml"
            }
            
            response = client.post("/dash/ingest", json=payload)
            assert response.status_code == 503
            
            # Test query endpoint
            query_payload = {
                "q": "test query",
                "top_k": 3
            }
            
            response = client.post("/dash/query", json=query_payload)
            assert response.status_code == 503
    
    def test_ingest_endpoint_job_failure(self, client, fixtures_path):
        """Test ingest endpoint when job fails."""
        with patch('app.dash_assistant.serving.routes.IndexJob') as mock_job_class:
            mock_job = AsyncMock()
            mock_job_class.return_value = mock_job
            
            # Mock job failure
            mock_job.load_dashboards_from_csv.side_effect = Exception("Database connection failed")
            
            with patch('app.dash_assistant.serving.routes.DashAssistantDB.health_check', return_value=True):
                payload = {
                    "dashboards_csv": str(fixtures_path / "dashboards.csv"),
                    "charts_csv": str(fixtures_path / "charts.csv"),
                    "md_dir": str(fixtures_path / "md"),
                    "enrichment_yaml": str(fixtures_path / "enrichment.yaml")
                }
                
                response = client.post("/dash/ingest", json=payload)
                
                assert response.status_code == 500
                data = response.json()
                assert "error" in data
                assert "Database connection failed" in data["error"]
    
    def test_query_endpoint_retriever_failure(self, client):
        """Test query endpoint when retriever fails."""
        with patch('app.dash_assistant.serving.routes.DashRetriever') as mock_retriever_class:
            mock_retriever = AsyncMock()
            mock_retriever_class.return_value = mock_retriever
            
            # Mock retriever failure
            mock_retriever.search.side_effect = Exception("Search index unavailable")
            
            with patch('app.dash_assistant.serving.routes.DashAssistantDB.health_check', return_value=True):
                payload = {
                    "q": "test query",
                    "top_k": 3
                }
                
                response = client.post("/dash/query", json=payload)
                
                assert response.status_code == 500
                data = response.json()
                assert "error" in data
                assert "Search index unavailable" in data["error"]


class TestDashAssistantModels:
    """Test request/response models for dash assistant routes."""
    
    def test_ingest_request_model(self):
        """Test IngestRequest model validation."""
        from app.dash_assistant.serving.routes import IngestRequest
        
        # Valid request
        valid_data = {
            "dashboards_csv": "/path/to/dashboards.csv",
            "charts_csv": "/path/to/charts.csv",
            "md_dir": "/path/to/md",
            "enrichment_yaml": "/path/to/enrichment.yaml",
            "run_embeddings": True
        }
        
        request = IngestRequest(**valid_data)
        assert request.dashboards_csv == "/path/to/dashboards.csv"
        assert request.run_embeddings is True
        
        # Test default value for run_embeddings
        minimal_data = {
            "dashboards_csv": "/path/to/dashboards.csv",
            "charts_csv": "/path/to/charts.csv",
            "md_dir": "/path/to/md",
            "enrichment_yaml": "/path/to/enrichment.yaml"
        }
        
        request = IngestRequest(**minimal_data)
        assert request.run_embeddings is False  # Default value
    
    def test_query_request_model(self):
        """Test QueryRequest model validation."""
        from app.dash_assistant.serving.routes import QueryRequest
        
        # Valid request
        valid_data = {
            "q": "revenue dashboard",
            "top_k": 5,
            "filters": {"domain": "finance"}
        }
        
        request = QueryRequest(**valid_data)
        assert request.q == "revenue dashboard"
        assert request.top_k == 5
        assert request.filters == {"domain": "finance"}
        
        # Test default values
        minimal_data = {
            "q": "test query"
        }
        
        request = QueryRequest(**minimal_data)
        assert request.q == "test query"
        assert request.top_k == 10  # Default value
        assert request.filters is None  # Default value
