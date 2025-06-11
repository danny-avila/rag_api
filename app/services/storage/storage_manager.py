"""
Storage manager with fallback and circuit breaker functionality
"""

import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages storage operations with fallback logic"""

    def __init__(
        self,
        primary_storage,
        fallback_storage=None,
        circuit_breaker_timeout: int = 300,
        failure_threshold: int = 3,
    ):
        """
        Initialize storage manager

        Args:
            primary_storage: Primary storage provider
            fallback_storage: Fallback storage provider (optional)
            circuit_breaker_timeout: Time in seconds before retrying primary after circuit opens
            failure_threshold: Number of failures before opening circuit
        """
        self.primary_storage = primary_storage
        self.fallback_storage = fallback_storage

        # Circuit breaker state
        self.primary_failures = 0
        self.circuit_open = False
        self.last_failure_time = None
        self.circuit_breaker_timeout = circuit_breaker_timeout
        self.failure_threshold = failure_threshold

    async def store_file(
        self,
        local_file_path: str,
        storage_key: str,
        content_type: Optional[str] = None,
        original_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Store file with automatic fallback

        Args:
            local_file_path: Path to the temporary file to store
            storage_key: Storage path (e.g., "agent123/document.pdf")
            content_type: MIME type of the file
            original_filename: Original filename before transformation

        Returns:
            Dictionary with storage metadata
        """
        # Check circuit breaker
        if self.circuit_open and self.last_failure_time:
            if time.time() - self.last_failure_time > self.circuit_breaker_timeout:
                self.circuit_open = False
                self.primary_failures = 0
                logger.info("Circuit breaker reset, retrying primary storage")

        # Try primary storage if circuit is closed
        if not self.circuit_open:
            try:
                result = await self.primary_storage.store_file(
                    local_file_path, storage_key, content_type, original_filename
                )
                self.primary_failures = 0  # Reset on success
                return result
            except Exception as e:
                logger.error(f"Primary storage failed: {e}")
                self.primary_failures += 1
                self.last_failure_time = time.time()

                if self.primary_failures >= self.failure_threshold:
                    self.circuit_open = True
                    logger.warning(
                        f"Primary storage circuit breaker opened after {self.failure_threshold} failures"
                    )

        # Use fallback if available
        if self.fallback_storage:
            logger.info("Using fallback storage")
            return await self.fallback_storage.store_file(
                local_file_path, storage_key, content_type, original_filename
            )

        # No fallback available
        raise Exception("All storage providers failed")

    async def delete_file(self, storage_key: str) -> bool:
        """
        Delete file from primary and fallback storage

        Args:
            storage_key: "agent123/document.pdf"

        Returns:
            True if deleted from at least one storage, False otherwise
        """
        deleted = False

        # Try to delete from primary
        try:
            if await self.primary_storage.delete_file(storage_key):
                deleted = True
        except Exception as e:
            logger.error(f"Failed to delete from primary storage: {e}")

        # Also try to delete from fallback
        if self.fallback_storage:
            try:
                if await self.fallback_storage.delete_file(storage_key):
                    deleted = True
            except Exception as e:
                logger.error(f"Failed to delete from fallback storage: {e}")

        return deleted

    def get_file_url(self, storage_key: str, expiration: int = 3600) -> Optional[str]:
        """
        Get file URL from primary storage (or fallback if primary is down)

        Args:
            storage_key: "agent123/document.pdf"
            expiration: URL expiration time in seconds

        Returns:
            File URL or path
        """
        # Check circuit breaker
        if not self.circuit_open:
            try:
                return self.primary_storage.get_file_url(storage_key, expiration)
            except Exception as e:
                logger.error(f"Failed to get URL from primary storage: {e}")

        # Use fallback if available
        if self.fallback_storage:
            return self.fallback_storage.get_file_url(storage_key, expiration)

        return None

    def get_folder_name(self, user_id: str, agent_id: Optional[str] = None) -> str:
        """
        Delegate to primary storage

        Args:
            user_id: User ID from authentication
            agent_id: Agent ID from request (optional)

        Returns:
            Folder name to use
        """
        return self.primary_storage.get_folder_name(user_id, agent_id)

    def generate_storage_key(
        self, folder_name: str, filename: str, file_id: str
    ) -> str:
        """
        Delegate to primary storage

        Args:
            folder_name: Folder name
            filename: Original filename
            file_id: Unique file identifier

        Returns:
            Storage key
        """
        return self.primary_storage.generate_storage_key(folder_name, filename, file_id)
