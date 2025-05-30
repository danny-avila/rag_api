"""
File Storage Service for RAG API
Handles both S3 and local storage with automatic fallback
"""

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import logging
import os
import shutil
import asyncio
import time
from typing import Optional, Dict, Any
import aiofiles
from datetime import datetime

logger = logging.getLogger(__name__)


class FileStorageService:
    """
    Unified file storage service that can store files in:
    1. AWS S3 (if S3_BUCKET_NAME is configured)
    2. Local persistent storage (fallback)

    Usage:
    service = FileStorageService(bucket_name="my-bucket")
    metadata = await service.store_file("temp.pdf", "agent123/document.pdf", "application/pdf")
    """

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        region: Optional[str] = None,
        local_storage_dir: str = "./storage/",
    ):
        """
        Initialize storage service

        Args:
            bucket_name: S3 bucket name (None = use local storage)
            region: AWS region for S3 (required for S3 usage)
            local_storage_dir: Directory for local storage fallback
        """
        self.bucket_name = bucket_name
        self.region = region
        self.local_storage_dir = local_storage_dir
        self.use_s3 = bucket_name is not None and region is not None

        # Circuit breaker for S3 failures
        self.s3_failure_count = 0
        self.s3_circuit_open = False
        self.last_failure_time = None

        # Initialize S3 client if bucket and region are provided
        if self.use_s3:
            try:
                self.s3_client = boto3.client("s3", region_name=region)
                logger.info(f"S3 client initialized for bucket: {bucket_name} in region: {region}")
            except Exception as e:
                logger.error(f"Failed to initialize S3 client: {e}")
                logger.info("Falling back to local storage")
                self.use_s3 = False
        elif bucket_name and not region:
            logger.warning(f"S3 bucket '{bucket_name}' specified but no region provided. Using local storage.")
        elif region and not bucket_name:
            logger.info(f"S3 region '{region}' specified but no bucket provided. Using local storage.")

        # Always ensure local storage directory exists (for temp files and fallback)
        os.makedirs(local_storage_dir, exist_ok=True)
        if not self.use_s3:
            logger.info(f"Local storage initialized: {local_storage_dir}")
        else:
            logger.info(f"Local storage directory created for fallback: {local_storage_dir}")

    def sanitize_path_component(self, component: str) -> str:
        """Sanitize agentID/userID/filename for safe storage"""
        import re

        # Remove path traversal attempts
        component = component.replace("..", "").replace("/", "").replace("\\", "")

        # Replace problematic characters
        component = re.sub(r'[<>:"|?*]', "_", component)

        # Limit length (S3 key max is 1024 chars, folder + filename)
        if len(component) > 100:
            name, ext = os.path.splitext(component)
            component = name[:95] + ext

        return component

    async def store_file(
        self, local_file_path: str, storage_key: str, content_type: str = None
    ) -> Dict[str, Any]:
        """
        Store file in S3 or local storage and return metadata

        Args:
            local_file_path: Path to the temporary file to store
            storage_key: Storage path (e.g., "agent123/document.pdf")
            content_type: MIME type of the file

        Returns:
            Dictionary with storage metadata

        Example:
            metadata = await store_file("/tmp/file.pdf", "agent123/doc.pdf", "application/pdf")
        """
        # Check circuit breaker for S3
        if self.s3_circuit_open and self.use_s3:
            if time.time() - self.last_failure_time > 300:  # 5 minutes
                self.s3_circuit_open = False
                self.s3_failure_count = 0
            else:
                logger.warning("S3 circuit breaker open, falling back to local storage")
                return await self._save_locally(
                    local_file_path, storage_key, content_type
                )

        try:
            if self.use_s3:
                result = await self._upload_to_s3(
                    local_file_path, storage_key, content_type
                )
                # Reset failure count on success
                self.s3_failure_count = 0
                return result
            else:
                return await self._save_locally(
                    local_file_path, storage_key, content_type
                )
        except Exception as e:
            if self.use_s3:
                self.s3_failure_count += 1
                self.last_failure_time = time.time()

                # Open circuit after 3 failures
                if self.s3_failure_count >= 3:
                    self.s3_circuit_open = True
                    logger.error("S3 circuit breaker opened due to repeated failures")

                # Fallback to local storage
                logger.warning(f"S3 upload failed, falling back to local: {e}")
                return await self._save_locally(
                    local_file_path, storage_key, content_type
                )
            else:
                logger.error(f"Storage operation failed: {e}")
                return None

    async def _upload_to_s3(
        self, local_file_path: str, s3_key: str, content_type: str = None
    ) -> Dict[str, Any]:
        """
        Upload file to S3 and return metadata

        This is where the actual S3 upload happens.
        """
        upload_params = None
        try:
            # Prepare upload parameters
            upload_params = {
                "Bucket": self.bucket_name,
                "Key": s3_key,
                "Body": open(local_file_path, "rb"),
            }

            if content_type:
                upload_params["ContentType"] = content_type

            # Upload to S3 (run in thread pool to avoid blocking)
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.s3_client.put_object(**upload_params)
            )

            # Get file info
            file_stats = os.stat(local_file_path)

            return {
                "storage_type": "s3",
                "bucket": self.bucket_name,
                "key": s3_key,
                "folder": s3_key.split("/")[0],
                "original_filename": os.path.basename(s3_key),
                "content_type": content_type or "application/octet-stream",
                "size_bytes": file_stats.st_size,
                "upload_timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            raise
        finally:
            # Always close file handle
            if upload_params and "Body" in upload_params:
                upload_params["Body"].close()

    async def _save_locally(
        self, temp_file_path: str, storage_key: str, content_type: str = None
    ) -> Dict[str, Any]:
        """
        Save file to local persistent storage and return metadata

        This copies the file from temp location to permanent storage.
        """
        # Build permanent path (e.g., "./storage/agent123/document.pdf")
        permanent_path = os.path.join(self.local_storage_dir, storage_key)

        # Create directory structure if it doesn't exist
        os.makedirs(os.path.dirname(permanent_path), exist_ok=True)

        # Copy file to permanent location (preserves metadata)
        shutil.copy2(temp_file_path, permanent_path)

        # Get file statistics
        file_stats = os.stat(permanent_path)

        return {
            "storage_type": "local",
            "path": permanent_path,
            "key": storage_key,
            "folder": storage_key.split("/")[0],
            "original_filename": os.path.basename(storage_key),
            "content_type": content_type or "application/octet-stream",
            "size_bytes": file_stats.st_size,
            "upload_timestamp": file_stats.st_mtime,
        }

    def generate_storage_key(
        self, folder_name: str, filename: str, file_id: str
    ) -> str:
        """
        Generate unique storage key with folder structure

        Args:
            folder_name: "agent123", "user456", or "public"
            filename: "document.pdf"
            file_id: Unique file identifier

        Returns:
            "agent123/document_file-123_20241201_143022.pdf"
        """
        # Sanitize inputs
        folder_name = self.sanitize_path_component(folder_name)
        filename = self.sanitize_path_component(filename)

        # Generate unique filename to prevent overwrites
        name, ext = os.path.splitext(filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{name}_{file_id[:8]}_{timestamp}{ext}"

        return f"{folder_name}/{unique_filename}"

    def get_folder_name(self, user_id: str, agent_id: Optional[str] = None) -> str:
        """
        Determine folder name based on agentID/userID priority

        Priority:
        1. agentID (if provided)
        2. userID (if not "public")
        3. "public" (fallback)

        Args:
            user_id: User ID from authentication
            agent_id: Agent ID from request (optional)

        Returns:
            Folder name to use
        """
        if agent_id:
            return agent_id
        elif user_id and user_id != "public":
            return user_id
        else:
            return "public"

    async def delete_file(self, storage_key: str) -> bool:
        """
        Delete file from S3 or local storage

        Args:
            storage_key: "agent123/document.pdf"

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            if self.use_s3:
                return await self._delete_from_s3(storage_key)
            else:
                return await self._delete_locally(storage_key)
        except Exception as e:
            logger.error(f"Delete operation failed for {storage_key}: {e}")
            return False

    async def _delete_from_s3(self, s3_key: str) -> bool:
        """Delete file from S3"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.delete_object(
                    Bucket=self.bucket_name, Key=s3_key
                ),
            )
            return True
        except Exception as e:
            logger.error(f"S3 delete failed: {e}")
            return False

    async def _delete_locally(self, storage_key: str) -> bool:
        """Delete file from local storage"""
        try:
            file_path = os.path.join(self.local_storage_dir, storage_key)
            if os.path.exists(file_path):
                os.remove(file_path)
            return True
        except Exception as e:
            logger.error(f"Local delete failed: {e}")
            return False

    def get_file_url(self, storage_key: str, expiration: int = 3600) -> str:
        """
        Generate presigned URL (S3) or local file path

        Args:
            storage_key: "agent123/document.pdf"
            expiration: URL expiration time in seconds

        Returns:
            Presigned URL (S3) or local file path
        """
        if self.use_s3:
            return self._get_presigned_url(storage_key, expiration)
        else:
            return os.path.join(self.local_storage_dir, storage_key)

    def _get_presigned_url(self, s3_key: str, expiration: int) -> str:
        """Generate presigned URL for S3 object"""
        try:
            return self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": s3_key},
                ExpiresIn=expiration,
            )
        except Exception as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            return None
