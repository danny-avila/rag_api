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
    primary_cooldown_minutes: int = Field(default=1, description="Minutes to wait before retrying failed primary provider")
    
    # State tracking (non-serializable)
    current_provider: Any = Field(default=None, init=False, exclude=True)
    current_provider_name: str = Field(default=None, init=False, exclude=True)
    consecutive_failures: int = Field(default=None, init=False, exclude=True)
    backup_success_count: int = Field(default=None, init=False, exclude=True)
    failover_lock: Any = Field(default=None, init=False, exclude=True)
    using_backup: bool = Field(default=None, init=False, exclude=True)
    primary_last_failure_time: Any = Field(default=None, init=False, exclude=True)
    
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
        self.primary_last_failure_time = None
        
        logger.info(
            f"Initialized BackupEmbeddingsProvider: "
            f"primary={self.primary_name}, backup={self.backup_name} "
            f"(cooldown={self.primary_cooldown_minutes}min after primary failure)"
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
    
    def _is_primary_in_cooldown(self) -> bool:
        """Check if primary provider is in cooldown period after recent failure."""
        if self.primary_last_failure_time is None:
            return False
            
        import time
        cooldown_seconds = self.primary_cooldown_minutes * 60
        time_since_failure = time.time() - self.primary_last_failure_time
        
        return time_since_failure < cooldown_seconds
    
    def _record_primary_failure(self):
        """Record primary provider failure with timestamp."""
        import time
        with self.failover_lock:
            self.primary_last_failure_time = time.time()
            self.consecutive_failures += 1
            if not self.using_backup:
                self.using_backup = True
                self.current_provider = self.backup_provider
                self.current_provider_name = self.backup_name
                logger.warning(f"🔄 Primary provider {self.primary_name} failed - switching to backup {self.backup_name} for {self.primary_cooldown_minutes} minutes")
    
    def _record_primary_recovery(self):
        """Record successful primary provider recovery."""
        with self.failover_lock:
            if self.using_backup:
                logger.info(f"✅ Primary provider {self.primary_name} recovered - switching back from backup")
            self.primary_last_failure_time = None
            self.consecutive_failures = 0
            self.using_backup = False
            self.backup_success_count = 0
            self.current_provider = self.primary_provider
            self.current_provider_name = self.primary_name
    
    def _embed_with_failover(self, texts: List[str]) -> List[List[float]]:
        """Embed texts with immediate failover on failure."""
        
        # Check if primary is available (not in cooldown)
        if not self._is_primary_in_cooldown():
            # Try primary provider
            try:
                logger.debug(f"Trying primary provider {self.primary_name} for {len(texts)} texts")
                result = self.primary_provider.embed_documents(texts)
                
                # Primary succeeded!
                self._record_primary_recovery()
                self._handle_provider_success()
                return result
                
            except Exception as primary_error:
                # Primary failed - log it prominently and immediately try backup
                logger.warning(f"❌ PRIMARY PROVIDER FAILED: {self.primary_name} - {str(primary_error)}")
                self._record_primary_failure()
                
                # Immediately try backup provider
                try:
                    logger.info(f"🔄 Immediately trying backup provider {self.backup_name} after primary failure")
                    result = self.backup_provider.embed_documents(texts)
                    
                    # Backup succeeded! Update state and return success
                    with self.failover_lock:
                        if not self.using_backup:
                            self.current_provider = self.backup_provider
                            self.current_provider_name = self.backup_name
                            self.using_backup = True
                        self.backup_success_count += 1
                    
                    logger.info(f"✅ Backup provider {self.backup_name} succeeded after primary failure")
                    return result
                    
                except Exception as backup_error:
                    # Both providers failed - but check if primary might have recovered
                    logger.error(
                        f"❌ Both providers failed! "
                        f"Primary ({self.primary_name}): {str(primary_error)}, "
                        f"Backup ({self.backup_name}): {str(backup_error)}"
                    )
                    
                    # If primary failure was a connection issue, try a quick recovery check
                    if "port not listening" in str(primary_error) or "Connection refused" in str(primary_error):
                        logger.info("🔄 Quick recovery check - testing if primary service has come back online")
                        try:
                            # Quick test with single text
                            recovery_result = self.primary_provider.embed_documents(["test"])
                            if recovery_result:
                                logger.info(f"✅ Primary provider {self.primary_name} has recovered! Using for current request")
                                self._record_primary_recovery()
                                # Retry the original request with recovered primary
                                return self.primary_provider.embed_documents(texts)
                        except Exception as recovery_error:
                            logger.debug(f"Primary still unavailable: {str(recovery_error)[:100]}")
                    
                    raise RuntimeError(
                        f"All embedding providers failed. Primary ({self.primary_name}): {str(primary_error)}, "
                        f"Backup ({self.backup_name}): {str(backup_error)}"
                    ) from backup_error
        else:
            # Primary is in cooldown - use backup directly
            remaining_cooldown = self.primary_cooldown_minutes * 60 - (time.time() - self.primary_last_failure_time)
            logger.debug(f"Primary provider {self.primary_name} in cooldown ({remaining_cooldown/60:.1f} minutes remaining)")
            
            try:
                logger.debug(f"Using backup provider {self.backup_name} (primary in cooldown)")
                result = self.backup_provider.embed_documents(texts)
                
                # Backup succeeded!
                with self.failover_lock:
                    if not self.using_backup:
                        self.current_provider = self.backup_provider
                        self.current_provider_name = self.backup_name
                        self.using_backup = True
                    self.backup_success_count += 1
                
                return result
                
            except Exception as backup_error:
                logger.error(f"❌ Backup provider failed while primary in cooldown: {str(backup_error)}")
                raise RuntimeError(f"Backup embedding failed (primary in cooldown): {str(backup_error)}") from backup_error
    
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