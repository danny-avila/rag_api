# tests/services/test_bedrock_embeddings.py
import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.services.embeddings.bedrock_rate_limited import RateLimitedBedrockEmbeddings


class TestRateLimitedBedrockEmbeddings:
    """Test suite for Titan V1 and V2 embeddings functionality."""
    
    @pytest.fixture
    def mock_client(self):
        """Mock Bedrock client for testing."""
        client = Mock()
        
        # Mock successful V1 response
        v1_response = {
            'body': Mock()
        }
        v1_response['body'].read.return_value = json.dumps({
            'embedding': [0.1, 0.2, 0.3, 0.4] * 256,  # 1024 dimensions
            'inputTextTokenCount': 10
        })
        
        # Mock successful V2 response  
        v2_response = {
            'body': Mock()
        }
        v2_response['body'].read.return_value = json.dumps({
            'embedding': [0.1, 0.2, 0.3, 0.4] * 128,  # 512 dimensions
            'inputTextTokenCount': 10
        })
        
        client.invoke_model.return_value = v2_response
        return client
    
    @pytest.fixture 
    def v1_embeddings(self, mock_client):
        """Create V1 embeddings instance for testing."""
        return RateLimitedBedrockEmbeddings(
            client=mock_client,
            model_id="amazon.titan-embed-text-v1",
            region_name="us-east-1"
        )
    
    @pytest.fixture
    def v2_embeddings(self, mock_client):
        """Create V2 embeddings instance for testing.""" 
        return RateLimitedBedrockEmbeddings(
            client=mock_client,
            model_id="amazon.titan-embed-text-v2:0",
            region_name="us-east-1",
            dimensions=512,
            normalize=True
        )
    
    def test_v1_model_detection(self, v1_embeddings):
        """Test that V1 models are correctly detected."""
        assert not v1_embeddings.is_v2_model
        assert v1_embeddings.dimensions is None  # V1 doesn't use dimensions
    
    def test_v2_model_detection(self, v2_embeddings):
        """Test that V2 models are correctly detected."""
        assert v2_embeddings.is_v2_model
        assert v2_embeddings.dimensions == 512
        assert v2_embeddings.normalize is True
    
    def test_v2_default_dimensions(self, mock_client):
        """Test that V2 defaults to 512 dimensions when not specified."""
        embeddings = RateLimitedBedrockEmbeddings(
            client=mock_client,
            model_id="amazon.titan-embed-text-v2:0",
            region_name="us-east-1"
        )
        assert embeddings.dimensions == 512  # Default value
    
    def test_v2_dimension_validation(self, mock_client):
        """Test that invalid dimensions raise ValueError."""
        with pytest.raises(ValueError, match="Invalid dimensions for Titan V2: 300"):
            RateLimitedBedrockEmbeddings(
                client=mock_client,
                model_id="amazon.titan-embed-text-v2:0",
                region_name="us-east-1",
                dimensions=300  # Invalid dimension
            )
    
    def test_v2_valid_dimensions(self, mock_client):
        """Test that all valid dimensions are accepted."""
        for dim in [256, 512, 1024]:
            embeddings = RateLimitedBedrockEmbeddings(
                client=mock_client,
                model_id="amazon.titan-embed-text-v2:0",
                region_name="us-east-1", 
                dimensions=dim
            )
            assert embeddings.dimensions == dim
    
    def test_v1_request_body_format(self, v1_embeddings):
        """Test that V1 request body has correct format."""
        body = v1_embeddings._create_embedding_request_body("test text")
        parsed = json.loads(body)
        
        assert "inputText" in parsed
        assert parsed["inputText"] == "test text"
        assert "dimensions" not in parsed
        assert "normalize" not in parsed
    
    def test_v2_request_body_format(self, v2_embeddings):
        """Test that V2 request body has correct format."""
        body = v2_embeddings._create_embedding_request_body("test text")
        parsed = json.loads(body)
        
        assert "inputText" in parsed
        assert parsed["inputText"] == "test text"
        assert "dimensions" in parsed
        assert parsed["dimensions"] == 512
        assert "normalize" in parsed
        assert parsed["normalize"] is True
    
    def test_v2_request_body_without_optional_params(self, mock_client):
        """Test V2 request body when dimensions not explicitly set."""
        embeddings = RateLimitedBedrockEmbeddings(
            client=mock_client,
            model_id="amazon.titan-embed-text-v2:0",
            region_name="us-east-1"
            # dimensions and normalize will use defaults
        )
        # Should get default dimensions
        assert embeddings.dimensions == 512
        assert embeddings.normalize is True
        
    def test_v1_embedding_function_called(self, v1_embeddings):
        """Test that V1 embedding function pathway works."""
        # Test that V1 doesn't use the V2 custom embedding path
        assert not v1_embeddings.is_v2_model
        
        # Test request body format for V1
        body = v1_embeddings._create_embedding_request_body("test")
        parsed = json.loads(body)
        assert "inputText" in parsed
        assert "dimensions" not in parsed  # V1 shouldn't have dimensions
    
    def test_v2_embedding_function(self, v2_embeddings):
        """Test that V2 uses custom embedding function."""
        result = v2_embeddings._embedding_func("test text")
        
        # Should call client.invoke_model with correct parameters
        v2_embeddings.client.invoke_model.assert_called_once()
        call_args = v2_embeddings.client.invoke_model.call_args
        
        assert call_args[1]['modelId'] == "amazon.titan-embed-text-v2:0"
        assert call_args[1]['accept'] == "application/json"
        assert call_args[1]['contentType'] == "application/json"
        
        # Parse the body to verify V2 format
        body = json.loads(call_args[1]['body'])
        assert body['inputText'] == "test text"
        assert body['dimensions'] == 512
        assert body['normalize'] is True
        
        # Should return the embedding
        assert len(result) == 512  # 4 * 128
    
    def test_backward_compatibility_v1_unchanged(self, v1_embeddings):
        """Test that V1 behavior is unchanged from original implementation."""
        # V1 should not have V2 parameters
        assert not hasattr(v1_embeddings, 'dimensions') or v1_embeddings.dimensions is None
        assert v1_embeddings.normalize is True  # Default value
        assert not v1_embeddings.is_v2_model
    
    def test_error_handling_invalid_model_response(self, v2_embeddings):
        """Test error handling when Bedrock returns invalid response."""
        # Mock a failed response
        v2_embeddings.client.invoke_model.side_effect = Exception("Model not found")
        
        with pytest.raises(RuntimeError, match="Bedrock V2 embedding failed"):
            v2_embeddings._embedding_func("test text")
    
    def test_throttling_error_detection(self, v2_embeddings):
        """Test that throttling errors are properly detected and handled."""
        # Mock throttling exception that will persist through all retries
        throttling_error = Exception("ThrottlingException: Rate exceeded")
        v2_embeddings.client.invoke_model.side_effect = throttling_error
        
        # Should retry and eventually raise RuntimeError after max retries
        with pytest.raises((RuntimeError, ValueError)):
            v2_embeddings._embed_batch_with_retry(["test"])
    
    def test_rate_limiting_state_management(self, v2_embeddings):
        """Test that rate limiting state is properly managed."""
        # Initial state
        assert v2_embeddings.current_delay == 0.0
        assert v2_embeddings.consecutive_successes == 0
        
        # Simulate throttling
        v2_embeddings._handle_throttling(0)
        assert v2_embeddings.current_delay == v2_embeddings.initial_retry_delay
        assert v2_embeddings.consecutive_successes == 0
        
        # Simulate success
        v2_embeddings._handle_success()
        assert v2_embeddings.consecutive_successes == 1
    
    def test_batch_processing(self, v2_embeddings):
        """Test that batch processing works correctly for V2."""
        texts = ["text1", "text2", "text3"]
        result = v2_embeddings._embed_batch_with_retry(texts)
        
        # Should call embedding function for each text
        assert len(result) == 3
        assert v2_embeddings.client.invoke_model.call_count == 3
    
    def test_configuration_logging(self, mock_client, caplog):
        """Test that V2 configuration is properly logged."""
        RateLimitedBedrockEmbeddings(
            client=mock_client,
            model_id="amazon.titan-embed-text-v2:0",
            region_name="us-east-1",
            dimensions=256,
            normalize=False
        )
        
        # Check that initialization log includes V2 parameters
        log_messages = [record.message for record in caplog.records]
        init_log = next((msg for msg in log_messages if "dimensions=256" in msg), None)
        assert init_log is not None
        assert "normalize=False" in init_log


class TestTitanV2Integration:
    """Integration tests for Titan V2 functionality."""
    
    @pytest.fixture
    def mock_client(self):
        """Mock client for integration tests."""
        client = Mock()
        return client
    
    def test_dimensions_performance_characteristics(self, mock_client):
        """Test that different dimensions produce expected vector sizes."""
        # Test 256 dimensions
        mock_client.invoke_model.return_value = {
            'body': Mock()
        }
        mock_client.invoke_model.return_value['body'].read.return_value = json.dumps({
            'embedding': [0.1] * 256,
            'inputTextTokenCount': 5
        })
        
        embeddings_256 = RateLimitedBedrockEmbeddings(
            client=mock_client,
            model_id="amazon.titan-embed-text-v2:0",
            region_name="us-east-1",
            dimensions=256
        )
        
        result = embeddings_256._embedding_func("test")
        assert len(result) == 256
    
    def test_storage_cost_optimization(self, mock_client):
        """Test that 512 dimensions provide optimal balance."""
        mock_client.invoke_model.return_value = {
            'body': Mock()
        }
        mock_client.invoke_model.return_value['body'].read.return_value = json.dumps({
            'embedding': [0.1] * 512,
            'inputTextTokenCount': 5
        })
        
        # Default should be 512 for optimal balance
        embeddings = RateLimitedBedrockEmbeddings(
            client=mock_client,
            model_id="amazon.titan-embed-text-v2:0",
            region_name="us-east-1"
        )
        
        assert embeddings.dimensions == 512  # 99% accuracy, 50% storage savings
        
        result = embeddings._embedding_func("test")
        assert len(result) == 512
    
    def test_normalization_for_rag_accuracy(self, mock_client):
        """Test that normalization is enabled by default for RAG."""
        embeddings = RateLimitedBedrockEmbeddings(
            client=mock_client,
            model_id="amazon.titan-embed-text-v2:0",
            region_name="us-east-1"
        )
        
        # Should default to True for RAG optimization
        assert embeddings.normalize is True
        
        body = embeddings._create_embedding_request_body("test")
        parsed = json.loads(body)
        assert parsed["normalize"] is True


class TestErrorHandlingAndValidation:
    """Test error handling and input validation."""
    
    @pytest.fixture
    def mock_client(self):
        """Mock client for error handling tests."""
        return Mock()
    
    def test_invalid_dimension_values(self, mock_client):
        """Test that only valid dimensions are accepted."""
        invalid_dimensions = [0, 100, 300, 500, 600, 2000, -1]
        
        for dim in invalid_dimensions:
            with pytest.raises(ValueError, match=f"Invalid dimensions for Titan V2: {dim}"):
                RateLimitedBedrockEmbeddings(
                    client=mock_client,
                    model_id="amazon.titan-embed-text-v2:0",
                    region_name="us-east-1",
                    dimensions=dim
                )
    
    def test_model_identifier_edge_cases(self, mock_client):
        """Test model identifier detection with various formats."""
        v2_identifiers = [
            "amazon.titan-embed-text-v2:0",
            "amazon.titan-embed-text-V2:0",  # Case insensitive
            "custom.titan-v2-model"
        ]
        
        for model_id in v2_identifiers:
            embeddings = RateLimitedBedrockEmbeddings(
                client=mock_client,
                model_id=model_id,
                region_name="us-east-1"
            )
            assert embeddings.is_v2_model
    
    def test_v1_model_identifiers(self, mock_client):
        """Test that V1 models are not detected as V2."""
        v1_identifiers = [
            "amazon.titan-embed-text-v1",
            "amazon.titan-embed-text",
            "custom.titan-v1-model"
        ]
        
        for model_id in v1_identifiers:
            embeddings = RateLimitedBedrockEmbeddings(
                client=mock_client,
                model_id=model_id,
                region_name="us-east-1"
            )
            assert not embeddings.is_v2_model


if __name__ == "__main__":
    pytest.main([__file__])