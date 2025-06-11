"""
Local file storage implementation
"""

import os
import shutil
import logging
from typing import Dict, Any, Optional
from .base_storage import BaseFileStorage

logger = logging.getLogger(__name__)


class LocalFileStorage(BaseFileStorage):
    """Local file storage implementation"""

    def __init__(self, storage_dir: str = "./storage/"):
        """
        Initialize local storage

        Args:
            storage_dir: Directory for local storage
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        logger.info(f"Local storage initialized: {storage_dir}")

    async def store_file(
        self,
        local_file_path: str,
        storage_key: str,
        content_type: Optional[str] = None,
        original_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Save file to local persistent storage and return metadata

        Args:
            local_file_path: Path to the temporary file to store
            storage_key: Storage path (e.g., "agent123/document.pdf")
            content_type: MIME type of the file
            original_filename: Original filename before transformation

        Returns:
            Dictionary with storage metadata
        """
        # Build permanent path (e.g., "./storage/agent123/document.pdf")
        permanent_path = os.path.join(self.storage_dir, storage_key)

        # Create directory structure if it doesn't exist
        os.makedirs(os.path.dirname(permanent_path), exist_ok=True)

        # Copy file to permanent location (preserves metadata)
        shutil.copy2(local_file_path, permanent_path)

        # Get file statistics
        file_stats = os.stat(permanent_path)

        return {
            "storage_type": "local",
            "path": permanent_path,
            "key": storage_key,
            "folder": storage_key.split("/")[0],
            "original_filename": original_filename or os.path.basename(storage_key),
            "content_type": content_type or "application/octet-stream",
            "size_bytes": file_stats.st_size,
            "upload_timestamp": file_stats.st_mtime,
        }

    async def delete_file(self, storage_key: str) -> bool:
        """
        Delete file from local storage

        Args:
            storage_key: "agent123/document.pdf"

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            file_path = os.path.join(self.storage_dir, storage_key)
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted local file: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Local delete failed for {storage_key}: {e}")
            return False

    def get_file_url(self, storage_key: str, expiration: int = 3600) -> str:
        """
        Get local file path (no URL for local storage)

        Args:
            storage_key: "agent123/document.pdf"
            expiration: Ignored for local storage

        Returns:
            Local file path
        """
        return os.path.join(self.storage_dir, storage_key)
