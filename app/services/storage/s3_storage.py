"""
S3 file storage implementation
"""

import os
import boto3
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from botocore.exceptions import ClientError, NoCredentialsError
from .base_storage import BaseFileStorage

logger = logging.getLogger(__name__)


class S3FileStorage(BaseFileStorage):
    """S3 file storage implementation"""

    def __init__(self, bucket_name: str, region: str):
        """
        Initialize S3 storage

        Args:
            bucket_name: S3 bucket name
            region: AWS region for S3
        """
        self.bucket_name = bucket_name
        self.region = region

        try:
            self.s3_client = boto3.client("s3", region_name=region)
            logger.info(
                f"S3 client initialized for bucket: {bucket_name} in region: {region}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise

    async def store_file(
        self,
        local_file_path: str,
        storage_key: str,
        content_type: Optional[str] = None,
        original_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Upload file to S3 and return metadata

        Args:
            local_file_path: Path to the temporary file to store
            storage_key: Storage path (e.g., "agent123/document.pdf")
            content_type: MIME type of the file
            original_filename: Original filename before transformation

        Returns:
            Dictionary with storage metadata
        """
        upload_params = None
        try:
            # Prepare upload parameters
            upload_params = {
                "Bucket": self.bucket_name,
                "Key": storage_key,
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
                "key": storage_key,
                "folder": storage_key.split("/")[0],
                "original_filename": original_filename or os.path.basename(storage_key),
                "content_type": content_type or "application/octet-stream",
                "size_bytes": file_stats.st_size,
                "upload_timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"S3 upload failed for {storage_key}: {e}")
            raise
        finally:
            # Always close file handle
            if upload_params and "Body" in upload_params:
                upload_params["Body"].close()

    async def delete_file(self, storage_key: str) -> bool:
        """
        Delete file from S3

        Args:
            storage_key: "agent123/document.pdf"

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.delete_object(
                    Bucket=self.bucket_name, Key=storage_key
                ),
            )
            logger.info(f"Deleted S3 object: s3://{self.bucket_name}/{storage_key}")
            return True
        except Exception as e:
            logger.error(f"S3 delete failed for {storage_key}: {e}")
            return False

    def get_file_url(self, storage_key: str, expiration: int = 3600) -> Optional[str]:
        """
        Generate presigned URL for S3 object

        Args:
            storage_key: "agent123/document.pdf"
            expiration: URL expiration time in seconds

        Returns:
            Presigned URL or None if generation fails
        """
        try:
            url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": storage_key},
                ExpiresIn=expiration,
            )
            return url
        except Exception as e:
            logger.error(f"Failed to generate presigned URL for {storage_key}: {e}")
            return None
