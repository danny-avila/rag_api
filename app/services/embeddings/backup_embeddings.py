"""Backup embeddings provider with automatic failover functionality."""
import time
from typing import List, Optional, Any
from threading import Lock
from pydantic import BaseModel, Field
from langchain_core.embeddings import Embeddings
from app.config import logger


class BackupEmbeddingsProvider(BaseModel, Embeddings):
    """
    Embeddings provider with automatic failover to backup provider.
    
    This wrapper:
    - Uses primary provider by default
    - Automatically fails over to backup on errors
    - Tracks provider health and switches back when primary recovers
    - Provides seamless fallback without service interruption
    """
    
    primary_provider: Any = Field(..., description="Primary embeddings provider")
    backup_provider: Any = Field(..., description="Backup embeddings provider") 
    primary_name: str = Field(..., description="Name of primary provider")
    backup_name: str = Field(..., description="Name of backup provider")
    
    # Failover configuration
    max_primary_failures: int = Field(default=3, description="Max consecutive failures before failover")
    recovery_check_interval: int = Field(default=10, description="Check primary recovery after N successful backup calls")
    
    # State tracking (non-serializable)
    current_provider: Any = Field(default=None, init=False, exclude=True)
    current_provider_name: str = Field(default=None, init=False, exclude=True)
    consecutive_failures: int = Field(default=None, init=False, exclude=True)
    backup_success_count: int = Field(default=None, init=False, exclude=True)
    failover_lock: Any = Field(default=None, init=False, exclude=True)
    using_backup: bool = Field(default=None, init=False, exclude=True)
    
    class Config:
        arbitrary_types_allowed = True
    
    def model_post_init(self, __context: Any) -> None:
        """Initialize failover state management."""
        # State management
        self.current_provider = self.primary_provider
        self.current_provider_name = self.primary_name
        self.consecutive_failures = 0
        self.backup_success_count = 0
        self.failover_lock = Lock()
        self.using_backup = False
        
        logger.info(
            f"Initialized BackupEmbeddingsProvider: "
            f"primary={self.primary_name}, backup={self.backup_name}, "
            f"max_failures={self.max_primary_failures}"
        )
    
    def _handle_provider_success(self):
        """Handle successful embedding call."""
        with self.failover_lock:
            if self.using_backup:
                self.backup_success_count += 1
                self.consecutive_failures = 0
            else:
                # Primary success - reset failure count
                self.consecutive_failures = 0
    
    def _handle_provider_failure(self, error: Exception) -> bool:
        """
        Handle provider failure and determine if failover should occur.
        
        Returns:
            bool: True if switched to backup, False if should retry with current provider
        """
        with self.failover_lock:
            if not self.using_backup:
                # Primary provider failed
                self.consecutive_failures += 1
                logger.warning(
                    f"{self.primary_name} failure {self.consecutive_failures}/{self.max_primary_failures}: {str(error)}"
                )
                
                if self.consecutive_failures >= self.max_primary_failures:
                    # Switch to backup
                    self.current_provider = self.backup_provider
                    self.current_provider_name = self.backup_name
                    self.using_backup = True
                    self.backup_success_count = 0
                    
                    logger.warning(
                        f"🔄 Failing over from {self.primary_name} to {self.backup_name} "
                        f"after {self.consecutive_failures} consecutive failures"
                    )
                    return True
            else:
                # Backup provider failed - this is more serious
                logger.error(
                    f"❌ BACKUP PROVIDER FAILED ({self.backup_name}): {str(error)}"
                )
                # Don't failover again, just propagate the error
            
            return False
    
    def _check_primary_recovery(self) -> bool:
        """Check if we should attempt to recover to primary provider."""
        with self.failover_lock:
            if (self.using_backup and 
                self.backup_success_count >= self.recovery_check_interval):
                
                logger.info(
                    f"🔍 Checking if {self.primary_name} has recovered "
                    f"(after {self.backup_success_count} successful backup calls)"
                )
                
                # Try a simple test call to primary
                try:
                    test_embedding = self.primary_provider.embed_query("test recovery")
                    if test_embedding:
                        # Primary is working again!
                        self.current_provider = self.primary_provider
                        self.current_provider_name = self.primary_name
                        self.using_backup = False
                        self.consecutive_failures = 0
                        self.backup_success_count = 0
                        
                        logger.info(f"✅ Recovered to primary provider: {self.primary_name}")
                        return True
                        
                except Exception as e:
                    logger.debug(f"Primary still failing: {str(e)}")
                    self.backup_success_count = 0  # Reset counter
                    
            return False
    
    def _embed_with_failover(self, texts: List[str]) -> List[List[float]]:
        """Embed texts with automatic failover support."""
        import time
        
        # Check if we should try to recover to primary
        self._check_primary_recovery()
        
        max_attempts = self.max_primary_failures if not self.using_backup else 1
        last_error = None
        
        # Try up to max_attempts times with the current provider
        for attempt in range(max_attempts):
            try:
                # Try current provider
                result = self.current_provider.embed_documents(texts)
                
                # Success!
                self._handle_provider_success()
                return result
                
            except Exception as e:
                last_error = e
                
                # If we're not at max failures yet, retry with backoff
                if attempt < max_attempts - 1:
                    backoff_delay = min(0.5 * (2 ** attempt), 5.0)
                    logger.info(
                        f"{self.current_provider_name} attempt {attempt + 1}/{max_attempts} failed: {str(e)[:100]}. "
                        f"Retrying in {backoff_delay:.1f}s..."
                    )
                    time.sleep(backoff_delay)
                    continue
                
                # We've exhausted retries with current provider
                logger.warning(
                    f"{self.current_provider_name} failed after {max_attempts} attempts: {str(e)}"
                )
        
        # If we get here, current provider failed all attempts
        # Update failure count
        if not self.using_backup:
            with self.failover_lock:
                self.consecutive_failures += max_attempts
        
        if not self.using_backup and self.consecutive_failures >= self.max_primary_failures:
            # Time to switch to backup
            with self.failover_lock:
                self.current_provider = self.backup_provider
                self.current_provider_name = self.backup_name
                self.using_backup = True
                self.backup_success_count = 0
            
            logger.warning(
                f"🔄 Failing over from {self.primary_name} to {self.backup_name} "
                f"after {self.consecutive_failures} consecutive failures"
            )
            
            # Try with backup provider
            try:
                logger.info(f"🔄 Using backup provider {self.backup_name}")
                result = self.current_provider.embed_documents(texts)
                self._handle_provider_success()
                
                # SUCCESS with backup - don't raise the primary error
                logger.warning(f"⚠️ Primary provider {self.primary_name} failed, but backup {self.backup_name} succeeded")
                logger.debug(f"Primary failure details: {str(last_error)}")
                
                return result
                
            except Exception as backup_error:
                logger.error(f"❌ Both providers failed! Primary: {str(last_error)}, Backup: {str(backup_error)}")
                raise RuntimeError(
                    f"All embedding providers failed. Primary ({self.primary_name}): {str(last_error)}, "
                    f"Backup ({self.backup_name}): {str(backup_error)}"
                ) from backup_error
        else:
            # Either backup failed or we haven't reached failover threshold
            raise RuntimeError(f"Embedding failed with {self.current_provider_name}: {str(last_error)}") from last_error
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents with backup failover."""
        if not texts:
            return []
        
        start_time = time.time()
        logger.debug(f"Embedding {len(texts)} texts using {self.current_provider_name}")
        
        try:
            result = self._embed_with_failover(texts)
            duration = time.time() - start_time
            
            provider_info = f"({self.current_provider_name})"
            if self.using_backup:
                provider_info = f"({self.current_provider_name} - backup active)"
                
            logger.info(f"Successfully embedded {len(texts)} texts {provider_info} in {duration:.2f}s")
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Failed to embed {len(texts)} texts after {duration:.2f}s: {str(e)}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query with backup failover."""
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