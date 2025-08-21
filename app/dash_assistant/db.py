# app/dash_assistant/db.py
"""Database connection module for dash assistant.

Uses dash_assistant specific configuration.
"""
import asyncpg
import structlog
from typing import Optional
from app.dash_assistant.config import get_config

# Setup logger for database operations
logger = structlog.get_logger(__name__)


class DashAssistantDB:
    """Database connection manager for dash assistant."""
    
    _pool: Optional[asyncpg.Pool] = None

    @classmethod
    async def get_pool(cls) -> asyncpg.Pool:
        """Get or create database connection pool.
        
        Reuses the same DSN configuration as the main project.
        
        Returns:
            asyncpg.Pool: Database connection pool
        """
        if cls._pool is None:
            try:
                config = get_config()
                dsn = config.database_url
                cls._pool = await asyncpg.create_pool(
                    dsn=dsn,
                    min_size=2,
                    max_size=10,
                    command_timeout=60
                )
                logger.info("Dash assistant database pool created", dsn=dsn)
            except Exception as e:
                logger.error(f"Failed to create dash assistant database pool: {e}")
                raise
        
        return cls._pool

    @classmethod
    async def close_pool(cls) -> None:
        """Close database connection pool."""
        if cls._pool is not None:
            await cls._pool.close()
            cls._pool = None
            logger.info("Dash assistant database pool closed")

    @classmethod
    async def health_check(cls) -> bool:
        """Check database connection health.
        
        Returns:
            bool: True if database is accessible, False otherwise
        """
        try:
            pool = await cls.get_pool()
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Dash assistant database health check failed: {e}")
            return False

    @classmethod
    async def execute_query(cls, query: str, *args) -> None:
        """Execute a query without returning results.
        
        Args:
            query: SQL query to execute
            *args: Query parameters
        """
        pool = await cls.get_pool()
        async with pool.acquire() as conn:
            await conn.execute(query, *args)

    @classmethod
    async def fetch_one(cls, query: str, *args) -> Optional[asyncpg.Record]:
        """Fetch a single record.
        
        Args:
            query: SQL query to execute
            *args: Query parameters
            
        Returns:
            Optional[asyncpg.Record]: Single record or None
        """
        pool = await cls.get_pool()
        async with pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    @classmethod
    async def fetch_all(cls, query: str, *args) -> list[asyncpg.Record]:
        """Fetch all records.
        
        Args:
            query: SQL query to execute
            *args: Query parameters
            
        Returns:
            list[asyncpg.Record]: List of records
        """
        pool = await cls.get_pool()
        async with pool.acquire() as conn:
            return await conn.fetch(query, *args)

    @classmethod
    async def fetch_value(cls, query: str, *args):
        """Fetch a single value.
        
        Args:
            query: SQL query to execute
            *args: Query parameters
            
        Returns:
            Any: Single value
        """
        pool = await cls.get_pool()
        async with pool.acquire() as conn:
            return await conn.fetchval(query, *args)


# Convenience functions for common operations
async def get_db_pool() -> asyncpg.Pool:
    """Get database connection pool."""
    return await DashAssistantDB.get_pool()


async def close_db_pool() -> None:
    """Close database connection pool."""
    await DashAssistantDB.close_pool()


async def db_health_check() -> bool:
    """Check database health."""
    return await DashAssistantDB.health_check()
