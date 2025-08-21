# app/dash_assistant/migrate.py
"""Simple migration runner for dash assistant.

Executes SQL migration files in alphabetical order.
"""
import os
import sys
import asyncio
import structlog
from pathlib import Path
from typing import List
from app.dash_assistant.db import DashAssistantDB

# Setup logger for migration
logger = structlog.get_logger(__name__)


class MigrationRunner:
    """Simple migration runner that executes SQL files in order."""
    
    def __init__(self, migrations_dir: str = None):
        """Initialize migration runner.
        
        Args:
            migrations_dir: Directory containing migration files
        """
        if migrations_dir is None:
            # Default to migrations directory relative to this file
            current_dir = Path(__file__).parent
            migrations_dir = current_dir / "migrations"
        
        self.migrations_dir = Path(migrations_dir)
        
        if not self.migrations_dir.exists():
            raise FileNotFoundError(f"Migrations directory not found: {self.migrations_dir}")

    def get_migration_files(self) -> List[Path]:
        """Get all SQL migration files sorted alphabetically.
        
        Returns:
            List[Path]: List of migration file paths
        """
        sql_files = list(self.migrations_dir.glob("*.sql"))
        return sorted(sql_files)

    async def create_migrations_table(self) -> None:
        """Create migrations tracking table if it doesn't exist."""
        await DashAssistantDB.execute_query("""
            CREATE TABLE IF NOT EXISTS dash_migrations (
                id SERIAL PRIMARY KEY,
                filename TEXT NOT NULL UNIQUE,
                applied_at TIMESTAMPTZ DEFAULT now()
            )
        """)

    async def get_applied_migrations(self) -> set[str]:
        """Get set of already applied migration filenames.
        
        Returns:
            set[str]: Set of applied migration filenames
        """
        try:
            records = await DashAssistantDB.fetch_all(
                "SELECT filename FROM dash_migrations ORDER BY id"
            )
            return {record['filename'] for record in records}
        except Exception:
            # Table might not exist yet
            return set()

    async def mark_migration_applied(self, filename: str) -> None:
        """Mark migration as applied.
        
        Args:
            filename: Migration filename to mark as applied
        """
        await DashAssistantDB.execute_query(
            "INSERT INTO dash_migrations (filename) VALUES ($1)",
            filename
        )

    async def execute_migration_file(self, file_path: Path) -> None:
        """Execute a single migration file.
        
        Args:
            file_path: Path to migration file
        """
        logger.info(f"Executing migration: {file_path.name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # Execute the entire file as one transaction
            # This handles dollar-quoted strings and complex SQL properly
            await DashAssistantDB.execute_query(sql_content)
            
            await self.mark_migration_applied(file_path.name)
            logger.info(f"Migration completed: {file_path.name}")
            
        except Exception as e:
            logger.error(f"Migration failed: {file_path.name} - {e}")
            raise

    async def run_migrations(self, dry_run: bool = False) -> None:
        """Run all pending migrations.
        
        Args:
            dry_run: If True, only show what would be executed
        """
        logger.info("Starting dash assistant migrations")
        
        # Ensure database connection
        await DashAssistantDB.get_pool()
        
        # Create migrations table
        if not dry_run:
            await self.create_migrations_table()
        
        # Get migration files and applied migrations
        migration_files = self.get_migration_files()
        applied_migrations = await self.get_applied_migrations() if not dry_run else set()
        
        if not migration_files:
            logger.info("No migration files found")
            return
        
        pending_migrations = [
            f for f in migration_files 
            if f.name not in applied_migrations
        ]
        
        if not pending_migrations:
            logger.info("No pending migrations")
            return
        
        logger.info(f"Found {len(pending_migrations)} pending migrations")
        
        for migration_file in pending_migrations:
            if dry_run:
                logger.info(f"Would execute: {migration_file.name}")
            else:
                await self.execute_migration_file(migration_file)
        
        if dry_run:
            logger.info("Dry run completed")
        else:
            logger.info("All migrations completed successfully")

    async def rollback_last_migration(self) -> None:
        """Rollback the last applied migration.
        
        Note: This is a simple implementation that only removes the record
        from the migrations table. Actual rollback SQL would need to be
        implemented separately.
        """
        last_migration = await DashAssistantDB.fetch_one(
            "SELECT filename FROM dash_migrations ORDER BY id DESC LIMIT 1"
        )
        
        if not last_migration:
            logger.info("No migrations to rollback")
            return
        
        filename = last_migration['filename']
        await DashAssistantDB.execute_query(
            "DELETE FROM dash_migrations WHERE filename = $1",
            filename
        )
        
        logger.info(f"Rolled back migration record: {filename}")
        logger.warning("Note: This only removes the migration record. "
                      "Manual rollback of schema changes may be required.")

    async def list_migrations(self) -> None:
        """List all migrations and their status."""
        migration_files = self.get_migration_files()
        applied_migrations = await self.get_applied_migrations()
        
        logger.info("Migration status:")
        for migration_file in migration_files:
            status = "APPLIED" if migration_file.name in applied_migrations else "PENDING"
            logger.info(f"  {migration_file.name}: {status}")


async def main():
    """Main entry point for migration runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dash Assistant Migration Runner")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be executed without running")
    parser.add_argument("--rollback", action="store_true",
                       help="Rollback the last migration")
    parser.add_argument("--list", action="store_true",
                       help="List all migrations and their status")
    parser.add_argument("--migrations-dir", type=str,
                       help="Custom migrations directory path")
    
    args = parser.parse_args()
    
    try:
        runner = MigrationRunner(args.migrations_dir)
        
        if args.list:
            await runner.list_migrations()
        elif args.rollback:
            await runner.rollback_last_migration()
        else:
            await runner.run_migrations(dry_run=args.dry_run)
            
    except Exception as e:
        logger.error(f"Migration runner failed: {e}")
        sys.exit(1)
    finally:
        await DashAssistantDB.close_pool()


if __name__ == "__main__":
    asyncio.run(main())
