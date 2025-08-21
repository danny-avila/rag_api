"""
Test configuration management for dash_assistant.

Tests both default values (without ENV) and environment variable overrides.
"""
import os
import pytest
from unittest.mock import patch

from app.dash_assistant.config import DashAssistantConfig


class TestDashAssistantConfig:
    """Test configuration loading and defaults."""

    def test_default_values_without_env(self):
        """Test default configuration values when no environment variables are set."""
        # Clear any existing environment variables
        env_vars_to_clear = [
            "EMBEDDINGS_PROVIDER",
            "EMBEDDINGS_DIMENSION", 
            "RRF_K",
            "DEFAULT_TOPK",
            "OPENAI_API_KEY",
            "DATABASE_URL",
            "POSTGRES_HOST",
            "POSTGRES_PORT",
            "POSTGRES_DB",
            "POSTGRES_USER",
            "POSTGRES_PASSWORD",
        ]
        
        with patch.dict(os.environ, {}, clear=True):
            config = DashAssistantConfig()
            
            # Test embedding defaults
            assert config.embeddings_provider == "MOCK"
            assert config.embeddings_dimension == 3072
            
            # Test retrieval defaults
            assert config.rrf_k == 60
            assert config.default_topk == 5
            
            # Test database defaults
            assert config.postgres_host == "localhost"
            assert config.postgres_port == 5432
            assert config.postgres_db == "rag_api"
            assert config.postgres_user == "postgres"
            assert config.postgres_password == "password"
            
            # Test optional fields
            assert config.openai_api_key is None

    def test_environment_variable_overrides(self):
        """Test that environment variables properly override defaults."""
        env_overrides = {
            "EMBEDDINGS_PROVIDER": "OPENAI",
            "EMBEDDINGS_DIMENSION": "1536",
            "RRF_K": "100", 
            "DEFAULT_TOPK": "10",
            "OPENAI_API_KEY": "test-api-key",
            "POSTGRES_HOST": "test-host",
            "POSTGRES_PORT": "5433",
            "POSTGRES_DB": "test_db",
            "POSTGRES_USER": "test_user",
            "POSTGRES_PASSWORD": "test_password",
        }
        
        with patch.dict(os.environ, env_overrides, clear=True):
            config = DashAssistantConfig()
            
            # Test embedding overrides
            assert config.embeddings_provider == "OPENAI"
            assert config.embeddings_dimension == 1536
            
            # Test retrieval overrides
            assert config.rrf_k == 100
            assert config.default_topk == 10
            
            # Test database overrides
            assert config.postgres_host == "test-host"
            assert config.postgres_port == 5433
            assert config.postgres_db == "test_db"
            assert config.postgres_user == "test_user"
            assert config.postgres_password == "test_password"
            
            # Test optional field override
            assert config.openai_api_key == "test-api-key"

    def test_database_url_property(self):
        """Test that database_url property is constructed correctly."""
        with patch.dict(os.environ, {}, clear=True):
            config = DashAssistantConfig()
            expected_url = "postgresql://postgres:password@localhost:5432/rag_api"
            assert config.database_url == expected_url

    def test_database_url_with_custom_values(self):
        """Test database_url with custom environment values."""
        env_overrides = {
            "POSTGRES_HOST": "custom-host",
            "POSTGRES_PORT": "5433",
            "POSTGRES_DB": "custom_db",
            "POSTGRES_USER": "custom_user",
            "POSTGRES_PASSWORD": "custom_pass",
        }
        
        with patch.dict(os.environ, env_overrides, clear=True):
            config = DashAssistantConfig()
            expected_url = "postgresql://custom_user:custom_pass@custom-host:5433/custom_db"
            assert config.database_url == expected_url

    def test_embeddings_provider_validation(self):
        """Test that embeddings_provider accepts valid values."""
        valid_providers = ["MOCK", "OPENAI", "HUGGINGFACE"]
        
        for provider in valid_providers:
            with patch.dict(os.environ, {"EMBEDDINGS_PROVIDER": provider}, clear=True):
                config = DashAssistantConfig()
                assert config.embeddings_provider == provider

    def test_positive_integer_validation(self):
        """Test that positive integer fields validate correctly."""
        # Test valid positive integers
        with patch.dict(os.environ, {
            "EMBEDDINGS_DIMENSION": "1536",
            "RRF_K": "60", 
            "DEFAULT_TOPK": "5",
            "POSTGRES_PORT": "5432"
        }, clear=True):
            config = DashAssistantConfig()
            assert config.embeddings_dimension == 1536
            assert config.rrf_k == 60
            assert config.default_topk == 5
            assert config.postgres_port == 5432

    def test_config_singleton_behavior(self):
        """Test that config behaves consistently across multiple instantiations."""
        with patch.dict(os.environ, {"EMBEDDINGS_PROVIDER": "OPENAI"}, clear=True):
            config1 = DashAssistantConfig()
            config2 = DashAssistantConfig()
            
            assert config1.embeddings_provider == config2.embeddings_provider
            assert config1.embeddings_dimension == config2.embeddings_dimension

    def test_deterministic_mock_config(self):
        """Test that MOCK provider config is deterministic for tests."""
        with patch.dict(os.environ, {}, clear=True):
            config = DashAssistantConfig()
            
            # These values should be deterministic for testing
            assert config.embeddings_provider == "MOCK"
            assert config.embeddings_dimension == 3072
            assert config.rrf_k == 60
            assert config.default_topk == 5
            
            # Should be consistent across multiple instantiations
            config2 = DashAssistantConfig()
            assert config.embeddings_provider == config2.embeddings_provider
            assert config.embeddings_dimension == config2.embeddings_dimension
