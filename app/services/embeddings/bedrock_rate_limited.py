"""Rate-limited Bedrock embeddings wrapper to handle throttling."""
import time
import asyncio
from typing import List
from threading import Semaphore, Lock
from langchain_aws import BedrockEmbeddings
from app.config import logger


class RateLimitedBedrockEmbeddings(BedrockEmbeddings):
    """
    Bedrock embeddings with rate limiting to prevent throttling errors.
    
    This wrapper adds:
    - Configurable rate limiting (requests per second)
    - Automatic retry with exponential backoff
    - Batch size control
    """
    
    def __init__(
        self, 
        *args, 
        max_requests_per_second: float = 2.0,
        max_batch_size: int = 10,
        max_retries: int = 5,
        initial_retry_delay: float = 1.0,
        **kwargs
    ):
        """
        Initialize rate-limited Bedrock embeddings.
        
        Args:
            max_requests_per_second: Maximum requests per second (default: 2.0)
            max_batch_size: Maximum texts to embed in one batch (default: 10)
            max_retries: Maximum number of retries on throttling (default: 5)
            initial_retry_delay: Initial delay in seconds for retry (default: 1.0)
        """
        super().__init__(*args, **kwargs)
        
        # Rate limiting
        self.max_requests_per_second = max_requests_per_second
        self.min_request_interval = 1.0 / max_requests_per_second
        self.last_request_time = 0
        self.request_lock = Lock()
        
        # Batch control
        self.max_batch_size = max_batch_size
        
        # Retry configuration
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        
        # Semaphore for concurrent request limiting
        self.semaphore = Semaphore(1)  # Process one batch at a time
        
        logger.info(
            f"Initialized RateLimitedBedrockEmbeddings: "
            f"max_rps={max_requests_per_second}, "
            f"batch_size={max_batch_size}, "
            f"max_retries={max_retries}"
        )
    
    def _wait_if_needed(self):
        """Wait if necessary to respect rate limit."""
        with self.request_lock:
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            
            if time_since_last_request < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last_request
                logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()
    
    def _embed_with_retry(self, text: str) -> List[float]:
        """
        Embed a single text with retry logic for throttling errors.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        retry_delay = self.initial_retry_delay
        
        for attempt in range(self.max_retries):
            try:
                # Wait to respect rate limit
                self._wait_if_needed()
                
                # Call parent's embedding function
                return super()._embedding_func(text)
                
            except Exception as e:
                error_message = str(e)
                
                # Check if it's a throttling error
                if "ThrottlingException" in error_message or "Too many requests" in error_message:
                    if attempt < self.max_retries - 1:
                        # Exponential backoff with jitter
                        jitter = retry_delay * 0.1 * (0.5 - time.time() % 1)
                        sleep_time = retry_delay + jitter
                        
                        logger.warning(
                            f"Throttling error on attempt {attempt + 1}/{self.max_retries}. "
                            f"Retrying in {sleep_time:.2f} seconds..."
                        )
                        
                        time.sleep(sleep_time)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        logger.error(f"Max retries ({self.max_retries}) exceeded for embedding")
                        raise
                else:
                    # Not a throttling error, raise immediately
                    raise
        
        raise Exception(f"Failed to embed text after {self.max_retries} attempts")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents with rate limiting and batch control.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        with self.semaphore:
            embeddings = []
            
            # Process in batches
            for i in range(0, len(texts), self.max_batch_size):
                batch = texts[i:i + self.max_batch_size]
                
                logger.debug(
                    f"Processing batch {i // self.max_batch_size + 1}: "
                    f"{len(batch)} texts"
                )
                
                # Embed each text in the batch with retry logic
                for text in batch:
                    embedding = self._embed_with_retry(text)
                    embeddings.append(embedding)
                
                # Additional delay between batches if we have more batches
                if i + self.max_batch_size < len(texts):
                    batch_delay = self.min_request_interval * 2  # Extra delay between batches
                    logger.debug(f"Batch complete, waiting {batch_delay:.2f}s before next batch")
                    time.sleep(batch_delay)
            
            logger.info(f"Successfully embedded {len(texts)} documents")
            return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query with rate limiting.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        with self.semaphore:
            return self._embed_with_retry(text)
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async version of embed_documents."""
        # Run synchronous method in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_documents, texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        """Async version of embed_query."""
        # Run synchronous method in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_query, text)