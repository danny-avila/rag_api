"""Custom NVIDIA embeddings provider for LLaMA embedding models."""
import json
import time
import requests
from typing import List, Optional, Any
from threading import Lock
from pydantic import Field, BaseModel
from langchain_core.embeddings import Embeddings
from app.config import logger


class NVIDIAEmbeddings(BaseModel, Embeddings):
    """
    Custom embeddings provider for NVIDIA LLaMA embedding models via OpenAI-compatible API.
    
    This provider handles the NVIDIA-specific API format requirements:
    - Array input format: "input": ["text1", "text2"]  
    - Required parameters: input_type, encoding_format, truncate
    - Batch processing with rate limiting
    """
    
    # Configuration fields
    base_url: str = Field(..., description="Base URL for the NVIDIA embedding endpoint")
    model: str = Field(..., description="NVIDIA model identifier")
    api_key: str = Field(default="dummy_key", description="API key (often not required for local endpoints)")
    max_batch_size: int = Field(default=20, description="Maximum texts to embed in one batch")
    max_retries: int = Field(default=3, description="Maximum retries on API errors")
    timeout: float = Field(default=30.0, description="Request timeout in seconds")
    input_type: str = Field(default="query", description="Input type for NVIDIA API")
    encoding_format: str = Field(default="float", description="Encoding format for embeddings")
    truncate: str = Field(default="NONE", description="Truncation behavior")
    
    # Rate limiting (reactive approach similar to Bedrock)
    initial_retry_delay: float = Field(default=0.1, description="Initial retry delay")
    max_retry_delay: float = Field(default=10.0, description="Maximum retry delay")
    backoff_factor: float = Field(default=1.5, description="Backoff multiplier")
    
    # Non-serializable fields
    current_delay: float = Field(default=None, init=False, exclude=True)
    delay_lock: Any = Field(default=None, init=False, exclude=True)
    consecutive_successes: int = Field(default=None, init=False, exclude=True)
    
    class Config:
        arbitrary_types_allowed = True
    
    def model_post_init(self, __context: Any) -> None:
        """Initialize non-serializable attributes after Pydantic model initialization."""
        # Rate limiting state
        self.current_delay = 0.0
        self.delay_lock = Lock()
        self.consecutive_successes = 0
        
        # Validate configuration
        if not self.base_url.startswith(('http://', 'https://')):
            raise ValueError(f"Invalid base_url: {self.base_url}")
        
        # Ensure URL ends with correct path
        if not self.base_url.endswith('/v1'):
            if self.base_url.endswith('/'):
                self.base_url += 'v1'
            else:
                self.base_url += '/v1'
        
        logger.info(
            f"Initialized NVIDIAEmbeddings: "
            f"model={self.model}, "
            f"base_url={self.base_url}, "
            f"batch_size={self.max_batch_size}, "
            f"input_type={self.input_type}"
        )
    
    def _apply_delay_if_needed(self):
        """Apply current delay if we're being rate limited."""
        with self.delay_lock:
            if self.current_delay > 0:
                logger.debug(f"Applying rate limit delay: {self.current_delay:.2f} seconds")
                time.sleep(self.current_delay)
    
    def _handle_success(self):
        """Reduce delay after successful call."""
        with self.delay_lock:
            if self.current_delay > 0:
                self.consecutive_successes += 1
                # Gradually reduce delay after consecutive successes
                if self.consecutive_successes >= 2:
                    self.current_delay *= 0.8  # Recovery factor
                    if self.current_delay < 0.1:
                        self.current_delay = 0.0
                        self.consecutive_successes = 0
                        logger.info("NVIDIA rate limiting removed - running at full speed")
    
    def _handle_error(self, attempt: int, error_message: str):
        """Handle errors and implement backoff if needed."""
        with self.delay_lock:
            # Check if it's a rate limiting error
            if any(err in error_message.lower() for err in [
                "rate limit", "too many requests", "429", "throttling"
            ]):
                self.consecutive_successes = 0
                if attempt == 0:
                    self.current_delay = self.initial_retry_delay
                else:
                    self.current_delay = min(
                        self.current_delay * self.backoff_factor,
                        self.max_retry_delay
                    )
                logger.warning(f"NVIDIA rate limited! Setting delay to {self.current_delay:.2f}s")
                return True  # Indicates this is a retryable error
            return False  # Not a rate limiting error
    
    def _create_embedding_request(self, texts: List[str]) -> dict:
        """Create the NVIDIA-specific embedding request payload."""
        return {
            "input": texts,  # Array format as required by NVIDIA
            "model": self.model,
            "input_type": self.input_type,
            "encoding_format": self.encoding_format,
            "truncate": self.truncate
        }
    
    def _embed_batch_with_retry(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts with retry logic for rate limiting and errors.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        request_data = self._create_embedding_request(texts)
        url = f"{self.base_url}/embeddings"
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Apply delay if we've been rate limited
                self._apply_delay_if_needed()
                
                logger.debug(f"Sending NVIDIA embedding request for {len(texts)} texts")
                
                response = requests.post(
                    url,
                    headers=headers,
                    json=request_data,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    # Success!
                    self._handle_success()
                    
                    result = response.json()
                    if "data" not in result:
                        raise ValueError(f"Invalid NVIDIA API response format: {result}")
                    
                    # Extract embeddings from response
                    embeddings = []
                    for item in result["data"]:
                        if "embedding" not in item:
                            raise ValueError(f"Missing embedding in response item: {item}")
                        embeddings.append(item["embedding"])
                    
                    logger.debug(f"Successfully embedded {len(embeddings)} texts")
                    return embeddings
                    
                elif response.status_code in [429, 503]:
                    # Rate limiting error
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    is_retryable = self._handle_error(attempt, error_msg)
                    
                    if is_retryable and attempt < self.max_retries - 1:
                        retry_delay = self.initial_retry_delay * (self.backoff_factor ** attempt)
                        retry_delay = min(retry_delay, self.max_retry_delay)
                        
                        logger.warning(
                            f"NVIDIA API rate limited on attempt {attempt + 1}/{self.max_retries}. "
                            f"Retrying in {retry_delay:.2f} seconds..."
                        )
                        time.sleep(retry_delay)
                        continue
                    else:
                        raise requests.RequestException(f"Max retries exceeded: {error_msg}")
                else:
                    # Other HTTP error
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    raise requests.RequestException(error_msg)
                    
            except requests.Timeout as e:
                last_error = e
                logger.warning(f"NVIDIA API timeout on attempt {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    time.sleep(1.0)  # Brief delay before retry
                    continue
                else:
                    raise TimeoutError(f"NVIDIA API timeout after {self.max_retries} attempts") from e
                    
            except requests.RequestException as e:
                last_error = e
                error_msg = str(e)
                
                # Check if it's retryable
                if self._handle_error(attempt, error_msg) and attempt < self.max_retries - 1:
                    retry_delay = self.initial_retry_delay * (self.backoff_factor ** attempt)
                    time.sleep(min(retry_delay, self.max_retry_delay))
                    continue
                else:
                    raise RuntimeError(f"NVIDIA embedding failed: {error_msg}") from e
            
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error in NVIDIA embedding: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(1.0)
                    continue
                else:
                    raise RuntimeError(f"NVIDIA embedding failed after {self.max_retries} attempts: {str(e)}") from e
        
        # Should not reach here
        raise RuntimeError(f"NVIDIA embedding failed after all retries: {str(last_error)}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents with batch processing and rate limiting.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        embeddings = []
        total_texts = len(texts)
        
        logger.info(f"Starting NVIDIA embedding for {total_texts} texts")
        start_time = time.time()
        
        # Process in batches
        for i in range(0, total_texts, self.max_batch_size):
            batch = texts[i:i + self.max_batch_size]
            batch_num = (i // self.max_batch_size) + 1
            total_batches = (total_texts + self.max_batch_size - 1) // self.max_batch_size
            
            logger.debug(f"Processing NVIDIA batch {batch_num}/{total_batches}: {len(batch)} texts")
            
            batch_embeddings = self._embed_batch_with_retry(batch)
            embeddings.extend(batch_embeddings)
        
        duration = time.time() - start_time
        logger.info(f"Completed NVIDIA embedding in {duration:.2f}s")
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        embeddings = self.embed_documents([text])
        return embeddings[0] if embeddings else []
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async version of embed_documents."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_documents, texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        """Async version of embed_query.""" 
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_query, text)