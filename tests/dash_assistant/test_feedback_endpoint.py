# tests/dash_assistant/test_feedback_endpoint.py
"""Tests for feedback endpoint functionality."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from main import app


class TestFeedbackEndpoint:
    """Test feedback endpoint functionality."""
    
    @pytest.fixture
    def client(self):
        """Create FastAPI test client."""
        return TestClient(app)
    
    def test_feedback_endpoint_success_up(self, client):
        """Test successful feedback submission with 'up' feedback."""
        with patch('app.dash_assistant.serving.routes.DashAssistantDB.health_check', return_value=True):
            with patch('app.dash_assistant.serving.routes.DashAssistantDB.execute_query') as mock_execute:
                mock_execute.return_value = None
                
                payload = {
                    "qid": 12345,
                    "entity_id": 1,
                    "chart_id": None,
                    "feedback": "up"
                }
                
                response = client.post("/dash/feedback", json=payload)
                
                assert response.status_code == 200
                data = response.json()
                
                assert "status" in data
                assert data["status"] == "success"
                assert "message" in data
                assert "Feedback recorded" in data["message"]
                
                # Verify database call
                mock_execute.assert_called_once()
                call_args = mock_execute.call_args
                assert "UPDATE query_log SET feedback = $1" in call_args[0][0]
                assert call_args[0][1] == "up"
                assert call_args[0][2] == 12345
    
    def test_feedback_endpoint_success_down(self, client):
        """Test successful feedback submission with 'down' feedback."""
        with patch('app.dash_assistant.serving.routes.DashAssistantDB.health_check', return_value=True):
            with patch('app.dash_assistant.serving.routes.DashAssistantDB.execute_query') as mock_execute:
                mock_execute.return_value = None
                
                payload = {
                    "qid": 67890,
                    "entity_id": 2,
                    "chart_id": 101,
                    "feedback": "down"
                }
                
                response = client.post("/dash/feedback", json=payload)
                
                assert response.status_code == 200
                data = response.json()
                
                assert data["status"] == "success"
                assert "Feedback recorded" in data["message"]
                
                # Verify database call
                mock_execute.assert_called_once()
                call_args = mock_execute.call_args
                assert call_args[0][1] == "down"
                assert call_args[0][2] == 67890
    
    def test_feedback_endpoint_validation_error_missing_qid(self, client):
        """Test feedback endpoint with missing qid."""
        payload = {
            "entity_id": 1,
            "feedback": "up"
        }
        
        response = client.post("/dash/feedback", json=payload)
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_feedback_endpoint_validation_error_invalid_feedback(self, client):
        """Test feedback endpoint with invalid feedback value."""
        payload = {
            "qid": 12345,
            "entity_id": 1,
            "feedback": "invalid"
        }
        
        response = client.post("/dash/feedback", json=payload)
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_feedback_endpoint_validation_error_negative_qid(self, client):
        """Test feedback endpoint with negative qid."""
        payload = {
            "qid": -1,
            "entity_id": 1,
            "feedback": "up"
        }
        
        response = client.post("/dash/feedback", json=payload)
        
        assert response.status_code == 422
    
    def test_feedback_endpoint_database_unavailable(self, client):
        """Test feedback endpoint when database is unavailable."""
        with patch('app.dash_assistant.serving.routes.DashAssistantDB.health_check', return_value=False):
            payload = {
                "qid": 12345,
                "entity_id": 1,
                "feedback": "up"
            }
            
            response = client.post("/dash/feedback", json=payload)
            
            assert response.status_code == 503
            data = response.json()
            assert "Database is not available" in data["detail"]
    
    def test_feedback_endpoint_database_error(self, client):
        """Test feedback endpoint when database operation fails."""
        with patch('app.dash_assistant.serving.routes.DashAssistantDB.health_check', return_value=True):
            with patch('app.dash_assistant.serving.routes.DashAssistantDB.execute_query') as mock_execute:
                mock_execute.side_effect = Exception("Database connection failed")
                
                payload = {
                    "qid": 12345,
                    "entity_id": 1,
                    "feedback": "up"
                }
                
                response = client.post("/dash/feedback", json=payload)
                
                assert response.status_code == 500
                data = response.json()
                assert "Feedback recording failed" in data["detail"]
    
    def test_feedback_endpoint_with_chart_id(self, client):
        """Test feedback endpoint with chart_id specified."""
        with patch('app.dash_assistant.serving.routes.DashAssistantDB.health_check', return_value=True):
            with patch('app.dash_assistant.serving.routes.DashAssistantDB.execute_query') as mock_execute:
                mock_execute.return_value = None
                
                payload = {
                    "qid": 12345,
                    "entity_id": 1,
                    "chart_id": 201,
                    "feedback": "up"
                }
                
                response = client.post("/dash/feedback", json=payload)
                
                assert response.status_code == 200
                
                # Verify database call includes chart_id update
                mock_execute.assert_called_once()
                call_args = mock_execute.call_args
                query = call_args[0][0]
                assert "chosen_chart = $3" in query
                assert call_args[0][3] == 201
    
    def test_feedback_endpoint_without_chart_id(self, client):
        """Test feedback endpoint without chart_id (None)."""
        with patch('app.dash_assistant.serving.routes.DashAssistantDB.health_check', return_value=True):
            with patch('app.dash_assistant.serving.routes.DashAssistantDB.execute_query') as mock_execute:
                mock_execute.return_value = None
                
                payload = {
                    "qid": 12345,
                    "entity_id": 1,
                    "chart_id": None,
                    "feedback": "down"
                }
                
                response = client.post("/dash/feedback", json=payload)
                
                assert response.status_code == 200
                
                # Verify database call
                mock_execute.assert_called_once()
                call_args = mock_execute.call_args
                query = call_args[0][0]
                assert "chosen_chart = $3" in query
                assert call_args[0][3] is None
    
    def test_feedback_endpoint_large_qid(self, client):
        """Test feedback endpoint with large qid value."""
        with patch('app.dash_assistant.serving.routes.DashAssistantDB.health_check', return_value=True):
            with patch('app.dash_assistant.serving.routes.DashAssistantDB.execute_query') as mock_execute:
                mock_execute.return_value = None
                
                payload = {
                    "qid": 9223372036854775807,  # Max int64
                    "entity_id": 1,
                    "feedback": "up"
                }
                
                response = client.post("/dash/feedback", json=payload)
                
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
    
    def test_feedback_endpoint_concurrent_requests(self, client):
        """Test feedback endpoint handles concurrent requests properly."""
        import threading
        import time
        
        results = []
        
        def submit_feedback(qid):
            with patch('app.dash_assistant.serving.routes.DashAssistantDB.health_check', return_value=True):
                with patch('app.dash_assistant.serving.routes.DashAssistantDB.execute_query') as mock_execute:
                    mock_execute.return_value = None
                    
                    payload = {
                        "qid": qid,
                        "entity_id": 1,
                        "feedback": "up"
                    }
                    
                    response = client.post("/dash/feedback", json=payload)
                    results.append(response.status_code)
        
        # Submit multiple feedback requests concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=submit_feedback, args=(i + 1,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert len(results) == 5
        assert all(status == 200 for status in results)


class TestFeedbackRequestModel:
    """Test FeedbackRequest Pydantic model."""
    
    def test_feedback_request_model_valid(self):
        """Test FeedbackRequest model with valid data."""
        from app.dash_assistant.serving.routes import FeedbackRequest
        
        # Valid request
        valid_data = {
            "qid": 12345,
            "entity_id": 1,
            "chart_id": 101,
            "feedback": "up"
        }
        
        request = FeedbackRequest(**valid_data)
        assert request.qid == 12345
        assert request.entity_id == 1
        assert request.chart_id == 101
        assert request.feedback == "up"
    
    def test_feedback_request_model_optional_chart_id(self):
        """Test FeedbackRequest model with optional chart_id."""
        from app.dash_assistant.serving.routes import FeedbackRequest
        
        # Valid request without chart_id
        valid_data = {
            "qid": 67890,
            "entity_id": 2,
            "feedback": "down"
        }
        
        request = FeedbackRequest(**valid_data)
        assert request.qid == 67890
        assert request.entity_id == 2
        assert request.chart_id is None  # Default value
        assert request.feedback == "down"
    
    def test_feedback_request_model_invalid_feedback(self):
        """Test FeedbackRequest model with invalid feedback value."""
        from app.dash_assistant.serving.routes import FeedbackRequest
        from pydantic import ValidationError
        
        invalid_data = {
            "qid": 12345,
            "entity_id": 1,
            "feedback": "invalid"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            FeedbackRequest(**invalid_data)
        
        assert "feedback" in str(exc_info.value)
    
    def test_feedback_request_model_negative_ids(self):
        """Test FeedbackRequest model with negative IDs."""
        from app.dash_assistant.serving.routes import FeedbackRequest
        from pydantic import ValidationError
        
        # Negative qid
        with pytest.raises(ValidationError):
            FeedbackRequest(qid=-1, entity_id=1, feedback="up")
        
        # Negative entity_id
        with pytest.raises(ValidationError):
            FeedbackRequest(qid=1, entity_id=-1, feedback="up")
        
        # Negative chart_id
        with pytest.raises(ValidationError):
            FeedbackRequest(qid=1, entity_id=1, chart_id=-1, feedback="up")
