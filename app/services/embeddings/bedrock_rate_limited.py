"""Reactive rate-limited Bedrock embeddings wrapper to handle throttling."""
import json
import time
import asyncio
from typing import List, Any, Optional
from threading import Lock
from pydantic import Field
from langchain_aws import BedrockEmbeddings
from app.config import logger


class RateLimitedBedrockEmbeddings(BedrockEmbeddings):
    """
    Bedrock embeddings with reactive rate limiting.
    
    This wrapper:
    - Runs at full speed until hitting AWS throttling
    - Implements exponential backoff when throttled
    - Gradually reduces wait time after successful calls
    - No unnecessary waiting when AWS can handle the load
    """
    
    # Declare Pydantic fields
    max_batch_size: int = Field(default=15, description="Maximum texts to embed in one batch")
    max_retries: int = Field(default=5, description="Maximum number of retries on throttling")
    initial_retry_delay: float = Field(default=0.1, description="Initial delay in seconds for retry")
    max_retry_delay: float = Field(default=30.0, description="Maximum delay between retries")
    backoff_factor: float = Field(default=2.0, description="Exponential backoff multiplier")
    recovery_factor: float = Field(default=0.9, description="Factor to reduce delay after success")
    
    # Titan V2 specific parameters
    dimensions: Optional[int] = Field(default=None, description="Output dimensions for V2 (256, 512, or 1024)")
    normalize: bool = Field(default=True, description="Whether to normalize embeddings (optimal for RAG)")
    
    # Non-serializable fields that will be set in model_post_init
    current_delay: float = Field(default=None, init=False, exclude=True)
    delay_lock: Any = Field(default=None, init=False, exclude=True)
    consecutive_successes: int = Field(default=None, init=False, exclude=True)
    is_v2_model: bool = Field(default=None, init=False, exclude=True)
    
    def model_post_init(self, __context: Any) -> None:
        """Initialize non-serializable attributes after Pydantic model initialization."""
        super().model_post_init(__context)
        
        # Reactive rate limiting state
        self.current_delay = 0.0  # Start with no delay
        self.delay_lock = Lock()
        self.consecutive_successes = 0
        
        # Detect model version and validate parameters
        self.is_v2_model = "v2" in self.model_id.lower() if hasattr(self, 'model_id') else False
        
        # Validate V2 parameters
        if self.is_v2_model:
            if self.dimensions is not None and self.dimensions not in [256, 512, 1024]:
                raise ValueError(f"Invalid dimensions for Titan V2: {self.dimensions}. Must be 256, 512, or 1024")
            # Set default dimensions for V2 if not specified
            if self.dimensions is None:
                self.dimensions = 512  # Sweet spot: 99% accuracy, 50% storage savings
                
        # Log initialization with V2 info
        v2_info = ""
        if self.is_v2_model:
            v2_info = f", dimensions={self.dimensions}, normalize={self.normalize}"
        
        logger.info(
            f"Initialized RateLimitedBedrockEmbeddings: "
            f"model={getattr(self, 'model_id', 'unknown')}, "
            f"batch_size={self.max_batch_size}, "
            f"max_retries={self.max_retries}, "
            f"reactive rate limiting enabled{v2_info}"
        )
    
    def _create_embedding_request_body(self, text: str) -> str:
        """Create the request body for embedding API call, handling V1/V2 differences."""
        if self.is_v2_model:
            # V2 format with dimensions and normalization
            body = {
                "inputText": text
            }
            if self.dimensions is not None:
                body["dimensions"] = self.dimensions
            if self.normalize is not None:
                body["normalize"] = self.normalize
            return json.dumps(body)
        else:
            # V1 format (simple text input)
            return json.dumps({"inputText": text})
    
    def _embedding_func(self, text: str) -> List[float]:
        """Override the embedding function to handle V1/V2 API differences."""
        if not self.is_v2_model:
            # Use parent's implementation for V1
            return super()._embedding_func(text)
        
        # Custom V2 implementation
        import boto3
        
        # Create the V2 request body
        body = self._create_embedding_request_body(text)
        
        # Make the API call
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body,
                accept="application/json",
                contentType="application/json"
            )
            
            # Parse the response
            response_body = json.loads(response['body'].read())
            return response_body['embedding']
            
        except Exception as e:
            # Re-raise with context
            raise RuntimeError(f"Bedrock V2 embedding failed: {str(e)}") from e
    
    def _apply_delay_if_needed(self):
        """Apply current delay if we're being throttled."""
        with self.delay_lock:
            if self.current_delay > 0:
                logger.debug(f"Applying reactive delay: {self.current_delay:.2f} seconds")
                time.sleep(self.current_delay)
    
    def _handle_success(self):
        """Reduce delay after successful call."""
        with self.delay_lock:
            if self.current_delay > 0:
                self.consecutive_successes += 1
                # Gradually reduce delay after consecutive successes
                if self.consecutive_successes >= 2:
                    self.current_delay *= self.recovery_factor
                    if self.current_delay < 0.1:
                        self.current_delay = 0.0
                        self.consecutive_successes = 0
                        logger.info("Rate limiting removed - running at full speed")
                    else:
                        logger.debug(f"Reducing delay to {self.current_delay:.2f}s after success")
    
    def _handle_throttling(self, retry_count: int):
        """Increase delay when throttled."""
        with self.delay_lock:
            self.consecutive_successes = 0
            if retry_count == 0:
                # First throttling, start with initial delay
                self.current_delay = self.initial_retry_delay
            else:
                # Exponential backoff
                self.current_delay = min(
                    self.current_delay * self.backoff_factor,
                    self.max_retry_delay
                )
            logger.warning(f"Throttled! Setting delay to {self.current_delay:.2f} seconds")
    
    def _embed_batch_with_retry(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts with reactive retry logic for throttling errors.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Apply delay only if we've been throttled before
                self._apply_delay_if_needed()
                
                # Try to embed the batch
                if hasattr(super(), 'embed_documents'):
                    result = super().embed_documents(texts)
                else:
                    # Fallback to single text embedding
                    result = []
                    for text in texts:
                        result.append(super()._embedding_func(text))
                
                # Success! Reduce delay for next time
                self._handle_success()
                return result
                
            except Exception as e:
                error_message = str(e)
                last_error = e
                
                # Check what type of error this is
                if any(err in error_message for err in [
                    "ThrottlingException", 
                    "Too many requests",
                    "Rate exceeded",
                    "TooManyRequestsException"
                ]):
                    # This is a throttling error - handle with backoff
                    self._handle_throttling(attempt)
                    
                    if attempt < self.max_retries - 1:
                        # Wait with exponential backoff
                        retry_delay = self.initial_retry_delay * (self.backoff_factor ** attempt)
                        retry_delay = min(retry_delay, self.max_retry_delay)
                        
                        logger.warning(
                            f"Throttling error on attempt {attempt + 1}/{self.max_retries}. "
                            f"Retrying in {retry_delay:.2f} seconds..."
                        )
                        
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"Max retries ({self.max_retries}) exceeded for embedding")
                        raise
                elif "ValidationException" in error_message and "model identifier is invalid" in error_message:
                    # This is a model configuration error - provide helpful message
                    logger.error(f"Invalid Bedrock model configuration: {error_message}")
                    from app.config import EMBEDDINGS_MODEL, AWS_DEFAULT_REGION
                    
                    helpful_message = (
                        f"❌ Bedrock Model Error: The model '{EMBEDDINGS_MODEL}' is not available.\n\n"
                        f"🔍 Possible causes:\n"
                        f"   • Model not available in region '{AWS_DEFAULT_REGION}'\n"
                        f"   • Model identifier is incorrect\n"
                        f"   • Model requires special access not enabled for your account\n\n"
                        f"💡 Try these solutions:\n"
                        f"   • Use 'amazon.titan-embed-text-v1' (widely available)\n"
                        f"   • Check AWS Bedrock console for available models in {AWS_DEFAULT_REGION}\n"
                        f"   • Verify your account has access to the model\n"
                        f"   • Set EMBEDDINGS_MODEL environment variable to a valid model"
                    )
                    
                    raise ValueError(helpful_message) from e
                elif "AccessDeniedException" in error_message:
                    # This is a permissions error
                    logger.error(f"Bedrock access denied: {error_message}")
                    from app.config import AWS_DEFAULT_REGION
                    
                    helpful_message = (
                        f"❌ Bedrock Access Denied: No permission to use Bedrock in '{AWS_DEFAULT_REGION}'.\n\n"
                        f"💡 Solutions:\n"
                        f"   • Enable Bedrock in AWS Console → Bedrock → Model access\n"
                        f"   • Request access to embedding models\n"
                        f"   • Verify IAM permissions include 'bedrock:InvokeModel'\n"
                        f"   • Check if your account is in the correct region"
                    )
                    
                    raise ValueError(helpful_message) from e
                else:
                    # Other non-throttling error, provide context and raise immediately
                    logger.error(f"Bedrock embedding error: {error_message}")
                    raise ValueError(f"Bedrock embedding failed: {error_message}") from e
        
        # Should not reach here, but just in case
        raise last_error if last_error else Exception(f"Failed to embed texts after {self.max_retries} attempts")
    
    def _embed_with_retry(self, text: str) -> List[float]:
        """
        Embed a single text with retry logic for throttling errors.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        return self._embed_batch_with_retry([text])[0]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents with reactive rate limiting.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        embeddings = []
        total_texts = len(texts)
        
        # Process in batches
        for i in range(0, total_texts, self.max_batch_size):
            batch = texts[i:i + self.max_batch_size]
            batch_num = (i // self.max_batch_size) + 1
            total_batches = (total_texts + self.max_batch_size - 1) // self.max_batch_size
            
            logger.debug(
                f"Processing batch {batch_num}/{total_batches}: "
                f"{len(batch)} texts"
            )
            
            # Process entire batch at once
            batch_embeddings = self._embed_batch_with_retry(batch)
            embeddings.extend(batch_embeddings)
        
        logger.info(f"Successfully embedded {len(texts)} documents")
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query with reactive rate limiting.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        return self._embed_with_retry(text)
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async version of embed_documents."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_documents, texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        """Async version of embed_query."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_query, text)