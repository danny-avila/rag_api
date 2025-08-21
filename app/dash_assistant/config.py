"""
Configuration management for dash_assistant.

Uses Pydantic BaseSettings for type-safe configuration with environment variable support.
Provides deterministic defaults for testing and production flexibility.
"""
from typing import Optional, Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class DashAssistantConfig(BaseSettings):
    """
    Configuration for dash_assistant module.
    
    Loads from environment variables with sensible defaults.
    Designed for deterministic testing (MOCK provider by default).
    """
    
    # Embeddings Configuration
    embeddings_provider: Literal["MOCK", "OPENAI", "HUGGINGFACE"] = Field(
        default="MOCK",
        alias="EMBEDDINGS_PROVIDER",
        description="Embeddings provider to use"
    )
    embeddings_dimension: int = Field(
        default=3072,
        alias="EMBEDDINGS_DIMENSION",
        gt=0,
        description="Dimension of embedding vectors"
    )
    
    # Retrieval Configuration  
    rrf_k: int = Field(
        default=60,
        alias="RRF_K",
        gt=0,
        description="RRF (Reciprocal Rank Fusion) k parameter"
    )
    default_topk: int = Field(
        default=5,
        alias="DEFAULT_TOPK",
        gt=0,
        description="Default number of results to return"
    )
    
    # Database Configuration
    postgres_host: str = Field(
        default="localhost",
        alias="POSTGRES_HOST",
        description="PostgreSQL host"
    )
    postgres_port: int = Field(
        default=5432,
        alias="POSTGRES_PORT",
        gt=0,
        le=65535,
        description="PostgreSQL port"
    )
    postgres_db: str = Field(
        default="rag_api",
        alias="POSTGRES_DB",
        description="PostgreSQL database name"
    )
    postgres_user: str = Field(
        default="postgres",
        alias="POSTGRES_USER",
        description="PostgreSQL username"
    )
    postgres_password: str = Field(
        default="password",
        alias="POSTGRES_PASSWORD",
        description="PostgreSQL password"
    )
    
    # Optional API Keys
    openai_api_key: Optional[str] = Field(
        default=None,
        alias="OPENAI_API_KEY",
        description="OpenAI API key for embeddings"
    )
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",  # Ignore extra environment variables
    }
    
    @property
    def database_url(self) -> str:
        """
        Construct PostgreSQL connection URL.
        
        Returns:
            Complete PostgreSQL connection string
        """
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    @field_validator("embeddings_dimension")
    @classmethod
    def validate_embeddings_dimension(cls, v: int) -> int:
        """Validate embeddings dimension is reasonable."""
        if v not in [384, 512, 768, 1024, 1536, 3072, 4096]:
            # Allow any positive integer but warn about common dimensions
            pass
        return v
    
    @field_validator("embeddings_provider", mode="before")
    @classmethod
    def validate_embeddings_provider(cls, v: str) -> str:
        """Validate embeddings provider."""
        return v.upper()


# Global configuration instance
# This ensures consistent configuration across the application
_config_instance: Optional[DashAssistantConfig] = None


def get_config() -> DashAssistantConfig:
    """
    Get the global configuration instance.
    
    Returns:
        DashAssistantConfig instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = DashAssistantConfig()
    return _config_instance


def reset_config() -> None:
    """
    Reset the global configuration instance.
    
    Useful for testing when you need to reload configuration.
    """
    global _config_instance
    _config_instance = None


# Convenience function for backward compatibility
def load_config() -> DashAssistantConfig:
    """
    Load configuration (alias for get_config).
    
    Returns:
        DashAssistantConfig instance
    """
    return get_config()
