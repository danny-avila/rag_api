# app/dash_assistant/indexing/embedder.py
"""Embeddings providers for dash assistant indexing."""
import hashlib
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union
import openai
from app.config import logger
from app.dash_assistant.config import get_config


class BaseEmbedder(ABC):
    """Base class for embedding providers."""
    
    def __init__(self, dimension: int = 1536):
        """Initialize embedder.
        
        Args:
            dimension: Embedding vector dimension
        """
        self.dimension = dimension
    
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            np.ndarray: Embedding vector
        """
        pass
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a batch of texts.
        
        Default implementation calls embed_text for each text.
        Subclasses can override for more efficient batch processing.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List[np.ndarray]: List of embedding vectors
        """
        return [self.embed_text(text) for text in texts]


class MockEmbedder(BaseEmbedder):
    """Mock embedder for testing with deterministic output."""
    
    def __init__(self, seed: int = 42, dimension: int = 3072):
        """Initialize mock embedder.
        
        Args:
            seed: Random seed for deterministic output
            dimension: Embedding vector dimension
        """
        super().__init__(dimension)
        self.seed = seed
        logger.info(f"Initialized MockEmbedder with seed={seed}, dimension={dimension}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate deterministic embedding for text.
        
        Uses hash of text content combined with seed to generate
        reproducible random vector.
        
        Args:
            text: Text to embed
            
        Returns:
            np.ndarray: Normalized embedding vector of specified dimension
        """
        # Create deterministic hash from text and seed
        text_hash = hashlib.sha256(f"{self.seed}:{text}".encode()).hexdigest()
        
        # Use hash as seed for numpy random generator
        hash_seed = int(text_hash[:8], 16) % (2**32)
        rng = np.random.RandomState(hash_seed)
        
        # Generate random vector
        vector = rng.normal(0, 1, self.dimension).astype(np.float32)
        
        # Normalize to unit length
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed batch of texts efficiently.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List[np.ndarray]: List of embedding vectors
        """
        return [self.embed_text(text) for text in texts]


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embeddings provider for production use."""
    
    def __init__(self, 
                 model: str = "text-embedding-3-small",
                 api_key: str = None,
                 base_url: str = None,
                 dimension: int = 1536):
        """Initialize OpenAI embedder.
        
        Args:
            model: OpenAI embedding model name
            api_key: OpenAI API key (defaults to env var)
            base_url: Custom API base URL
            dimension: Embedding dimension
        """
        super().__init__(dimension)
        self.model = model
        
        # Initialize OpenAI client
        client_kwargs = {}
        if api_key:
            client_kwargs['api_key'] = api_key
        if base_url:
            client_kwargs['base_url'] = base_url
            
        self.client = openai.OpenAI(**client_kwargs)
        
        logger.info(f"Initialized OpenAIEmbedder with model={model}, dimension={dimension}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed text using OpenAI API.
        
        Args:
            text: Text to embed
            
        Returns:
            np.ndarray: Embedding vector
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimension if self.model.startswith("text-embedding-3") else None
            )
            
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            return embedding
            
        except Exception as e:
            logger.error(f"OpenAI embedding failed for text: {text[:100]}... Error: {e}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed batch of texts using OpenAI API.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List[np.ndarray]: List of embedding vectors
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimension if self.model.startswith("text-embedding-3") else None
            )
            
            embeddings = [
                np.array(item.embedding, dtype=np.float32) 
                for item in response.data
            ]
            return embeddings
            
        except Exception as e:
            logger.error(f"OpenAI batch embedding failed for {len(texts)} texts. Error: {e}")
            raise


def get_embedder() -> BaseEmbedder:
    """Factory function to get embedder based on configuration.
    
    Returns:
        BaseEmbedder: Configured embedder instance
    """
    config = get_config()
    provider = config.embeddings_provider.lower()
    dimension = config.embeddings_dimension
    
    if provider == "mock":
        # Use fixed seed for deterministic testing
        seed = 42
        return MockEmbedder(seed=seed, dimension=dimension)
    
    elif provider == "openai":
        model = "text-embedding-3-small"  # Default OpenAI model
        api_key = config.openai_api_key
        
        return OpenAIEmbedder(
            model=model,
            api_key=api_key,
            dimension=dimension
        )
    
    else:
        raise ValueError(f"Unsupported embeddings provider: {provider}")
