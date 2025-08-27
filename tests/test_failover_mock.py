"""Mock tests for backup failover functionality."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.services.embeddings.backup_embeddings import BackupEmbeddingsProvider


class TestBackupFailover:
    """Test backup embedding provider failover behavior."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.primary_provider = Mock()
        self.backup_provider = Mock()
        
        self.backup_embeddings = BackupEmbeddingsProvider(
            primary_provider=self.primary_provider,
            backup_provider=self.backup_provider,
            primary_name="nvidia:test-model",
            backup_name="bedrock:test-model",
            primary_cooldown_minutes=1
        )
    
    def test_primary_success_no_failover(self):
        """Test successful primary provider - no failover needed."""
        # Arrange
        test_texts = ["test text 1", "test text 2"]
        expected_embeddings = [[0.1, 0.2], [0.3, 0.4]]
        self.primary_provider.embed_documents.return_value = expected_embeddings
        
        # Act
        result = self.backup_embeddings.embed_documents(test_texts)
        
        # Assert
        assert result == expected_embeddings
        self.primary_provider.embed_documents.assert_called_once_with(test_texts)
        self.backup_provider.embed_documents.assert_not_called()
    
    def test_primary_failure_immediate_backup_success(self):
        """Test primary failure triggers immediate backup attempt."""
        # Arrange
        test_texts = ["test text 1", "test text 2"]
        expected_embeddings = [[0.1, 0.2], [0.3, 0.4]]
        
        # Primary fails with connection error
        self.primary_provider.embed_documents.side_effect = TimeoutError("NVIDIA service not available")
        # Backup succeeds
        self.backup_provider.embed_documents.return_value = expected_embeddings
        
        # Act
        result = self.backup_embeddings.embed_documents(test_texts)
        
        # Assert
        assert result == expected_embeddings
        self.primary_provider.embed_documents.assert_called_once_with(test_texts)
        self.backup_provider.embed_documents.assert_called_once_with(test_texts)
    
    def test_both_providers_fail(self):
        """Test behavior when both providers fail."""
        # Arrange
        test_texts = ["test text 1", "test text 2"]
        
        # Both providers fail
        self.primary_provider.embed_documents.side_effect = TimeoutError("NVIDIA timeout")
        self.backup_provider.embed_documents.side_effect = RuntimeError("Bedrock error")
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="All embedding providers failed"):
            self.backup_embeddings.embed_documents(test_texts)
    
    def test_primary_cooldown_uses_backup_directly(self):
        """Test that backup is used directly when primary is in cooldown."""
        # Arrange
        test_texts = ["test text 1", "test text 2"]
        expected_embeddings = [[0.1, 0.2], [0.3, 0.4]]
        
        # Simulate primary in cooldown by setting failure time
        import time
        self.backup_embeddings.primary_last_failure_time = time.time()
        self.backup_embeddings.using_backup = True
        
        self.backup_provider.embed_documents.return_value = expected_embeddings
        
        # Act
        result = self.backup_embeddings.embed_documents(test_texts)
        
        # Assert
        assert result == expected_embeddings
        # Primary should not be called when in cooldown
        self.primary_provider.embed_documents.assert_not_called()
        self.backup_provider.embed_documents.assert_called_once_with(test_texts)
    
    @patch('app.services.embeddings.nvidia_embeddings.socket.socket')
    def test_nvidia_port_check_failure(self, mock_socket):
        """Test NVIDIA provider detects port not listening."""
        from app.services.embeddings.nvidia_embeddings import NVIDIAEmbeddings
        
        # Arrange - simulate port not listening
        mock_sock_instance = Mock()
        mock_socket.return_value = mock_sock_instance
        mock_sock_instance.connect_ex.return_value = 1  # Connection failed
        
        nvidia_provider = NVIDIAEmbeddings(
            base_url="http://localhost:8003/v1",
            model="test-model",
            api_key="test-key",
            timeout=3.0,
            max_retries=1  # Reduce retries for faster test
        )
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="NVIDIA service not available"):
            nvidia_provider.embed_documents(["test text"])
        
        # Verify socket check was attempted
        mock_socket.assert_called()
        mock_sock_instance.connect_ex.assert_called_with(('localhost', 8003))
    
    @patch('app.services.embeddings.nvidia_embeddings.socket.socket')
    @patch('app.services.embeddings.nvidia_embeddings.requests.post')
    def test_nvidia_port_check_success_but_request_fails(self, mock_post, mock_socket):
        """Test NVIDIA provider when port is listening but request fails."""
        from app.services.embeddings.nvidia_embeddings import NVIDIAEmbeddings
        
        # Arrange - simulate port listening but request fails
        mock_sock_instance = Mock()
        mock_socket.return_value = mock_sock_instance
        mock_sock_instance.connect_ex.return_value = 0  # Connection successful
        
        # Mock requests.post to fail with timeout
        mock_post.side_effect = TimeoutError("Read timeout")
        
        nvidia_provider = NVIDIAEmbeddings(
            base_url="http://localhost:8003/v1",
            model="test-model", 
            api_key="test-key",
            timeout=3.0,
            max_retries=1  # Reduce retries for faster test
        )
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="Read timeout"):
            nvidia_provider.embed_documents(["test text"])
        
        # Verify socket check passed and request was attempted
        mock_sock_instance.connect_ex.assert_called_with(('localhost', 8003))
        mock_post.assert_called_once()


class TestNVIDIASocketCheck:
    """Test NVIDIA provider socket check functionality."""
    
    @patch('app.services.embeddings.nvidia_embeddings.socket.socket')
    def test_port_available_check_success(self, mock_socket):
        """Test successful port availability check."""
        from app.services.embeddings.nvidia_embeddings import NVIDIAEmbeddings
        
        # Arrange
        mock_sock_instance = Mock()
        mock_socket.return_value = mock_sock_instance
        mock_sock_instance.connect_ex.return_value = 0  # Success
        
        nvidia_provider = NVIDIAEmbeddings(
            base_url="http://localhost:8003/v1",
            model="test-model",
            api_key="test-key"
        )
        
        # Act
        result = nvidia_provider._check_port_available("http://localhost:8003/v1")
        
        # Assert
        assert result is True
        mock_sock_instance.connect_ex.assert_called_with(('localhost', 8003))
        mock_sock_instance.close.assert_called_once()
    
    @patch('app.services.embeddings.nvidia_embeddings.socket.socket')
    def test_port_available_check_failure(self, mock_socket):
        """Test failed port availability check."""
        from app.services.embeddings.nvidia_embeddings import NVIDIAEmbeddings
        
        # Arrange
        mock_sock_instance = Mock()
        mock_socket.return_value = mock_sock_instance
        mock_sock_instance.connect_ex.return_value = 111  # Connection refused
        
        nvidia_provider = NVIDIAEmbeddings(
            base_url="http://localhost:8003/v1",
            model="test-model",
            api_key="test-key"
        )
        
        # Act
        result = nvidia_provider._check_port_available("http://localhost:8003/v1")
        
        # Assert
        assert result is False
        mock_sock_instance.connect_ex.assert_called_with(('localhost', 8003))
        mock_sock_instance.close.assert_called_once()