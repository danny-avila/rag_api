"""
Factory functions for creating storage providers
"""

import logging
from typing import Optional
from app.config import StorageProvider

logger = logging.getLogger(__name__)


def get_file_storage(provider: StorageProvider, **kwargs):
    """
    Factory function to create storage providers

    Args:
        provider: StorageProvider enum value
        **kwargs: Provider-specific parameters

    Returns:
        Storage provider instance
    """
    if provider == StorageProvider.S3:
        from .s3_storage import S3FileStorage

        bucket_name = kwargs.get("bucket_name")
        region = kwargs.get("region")

        if not bucket_name or not region:
            raise ValueError("S3 storage requires both bucket_name and region")

        return S3FileStorage(bucket_name=bucket_name, region=region)

    elif provider == StorageProvider.LOCAL:
        from .local_storage import LocalFileStorage

        storage_dir = kwargs.get("storage_dir", "./storage/")
        return LocalFileStorage(storage_dir=storage_dir)

    else:
        raise ValueError(f"Invalid storage provider: {provider}")


def init_storage_with_fallback(
    s3_bucket_name: Optional[str] = None,
    s3_region: Optional[str] = None,
    local_storage_dir: str = "./storage/",
    circuit_breaker_timeout: int = 300,
    failure_threshold: int = 3,
):
    """
    Initialize storage with automatic fallback

    Args:
        s3_bucket_name: S3 bucket name (optional)
        s3_region: AWS region (optional)
        local_storage_dir: Local storage directory
        circuit_breaker_timeout: Timeout for circuit breaker
        failure_threshold: Number of failures before opening circuit

    Returns:
        StorageManager instance with configured storage providers
    """
    from .storage_manager import StorageManager

    primary_storage = None
    fallback_storage = None

    # Try S3 if configured
    if s3_bucket_name and s3_region:
        try:
            primary_storage = get_file_storage(
                StorageProvider.S3, bucket_name=s3_bucket_name, region=s3_region
            )
            # Always create local fallback for S3
            fallback_storage = get_file_storage(
                StorageProvider.LOCAL, storage_dir=local_storage_dir
            )
            logger.info(
                f"Storage initialized with S3 primary (bucket: {s3_bucket_name}) and local fallback"
            )
        except Exception as e:
            logger.error(f"Failed to initialize S3, using local storage: {e}")
            primary_storage = get_file_storage(
                StorageProvider.LOCAL, storage_dir=local_storage_dir
            )
            logger.info(
                f"Storage initialized with local directory: {local_storage_dir}"
            )
    else:
        # Use local storage as primary
        primary_storage = get_file_storage(
            StorageProvider.LOCAL, storage_dir=local_storage_dir
        )
        logger.info(f"Storage initialized with local directory: {local_storage_dir}")

    return StorageManager(
        primary_storage,
        fallback_storage,
        circuit_breaker_timeout=circuit_breaker_timeout,
        failure_threshold=failure_threshold,
    )
