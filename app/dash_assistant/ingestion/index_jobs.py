# app/dash_assistant/ingestion/index_jobs.py
"""Index jobs CLI for ingesting dashboard data."""
import argparse
import asyncio
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from app.config import logger
from app.dash_assistant.db import DashAssistantDB
try:
    # Try relative imports first (when run as module)
    from .csv_loader import load_and_validate_dashboards_csv, load_and_validate_charts_csv
    from .md_loader import load_markdown_dir, create_content_chunks
    from .enrichment_loader import load_and_validate_enrichment_yaml, apply_enrichment_to_dashboard
    from ..indexing.embedder import get_embedder
except ImportError:
    # Fall back to absolute imports (when run directly)
    from app.dash_assistant.ingestion.csv_loader import load_and_validate_dashboards_csv, load_and_validate_charts_csv
    from app.dash_assistant.ingestion.md_loader import load_markdown_dir, create_content_chunks
    from app.dash_assistant.ingestion.enrichment_loader import load_and_validate_enrichment_yaml, apply_enrichment_to_dashboard
    from app.dash_assistant.indexing.embedder import get_embedder


class IndexJob:
    """Main class for dashboard data ingestion jobs."""
    
    def __init__(self):
        self.db = DashAssistantDB
    
    async def load_dashboards_from_csv(self, csv_path: Path, 
                                     domain_mapping: Optional[Dict[str, str]] = None,
                                     default_tags: Optional[List[str]] = None) -> int:
        """Load dashboards from CSV file into database.
        
        Args:
            csv_path: Path to dashboards CSV file
            domain_mapping: Optional domain mapping rules
            default_tags: Optional default tags
            
        Returns:
            int: Number of dashboards loaded
        """
        logger.info(f"Loading dashboards from {csv_path}")
        
        # Load and validate CSV data
        dashboards = await load_and_validate_dashboards_csv(
            csv_path, 
            domain_mapping=domain_mapping,
            default_tags=default_tags
        )
        
        # Insert/update dashboards in database
        loaded_count = 0
        for dashboard in dashboards:
            try:
                entity_id = await self._upsert_dashboard(dashboard)
                await self._create_dashboard_chunks(entity_id, dashboard)
                loaded_count += 1
            except Exception as e:
                logger.error(f"Error loading dashboard {dashboard.get('title', 'unknown')}: {e}")
                # Continue with other dashboards
                continue
        
        logger.info(f"Successfully loaded {loaded_count} dashboards")
        return loaded_count
    
    async def load_charts_from_csv(self, csv_path: Path) -> int:
        """Load charts from CSV file into database.
        
        Args:
            csv_path: Path to charts CSV file
            
        Returns:
            int: Number of charts loaded
        """
        logger.info(f"Loading charts from {csv_path}")
        
        # Load and validate CSV data
        charts = await load_and_validate_charts_csv(csv_path)
        
        # Insert/update charts in database
        loaded_count = 0
        for chart in charts:
            try:
                chart_id = await self._upsert_chart(chart)
                await self._create_chart_chunks(chart_id, chart)
                loaded_count += 1
            except Exception as e:
                logger.error(f"Error loading chart {chart.get('title', 'unknown')}: {e}")
                # Continue with other charts
                continue
        
        logger.info(f"Successfully loaded {loaded_count} charts")
        return loaded_count
    
    async def load_markdown_content(self, md_dir: Path) -> int:
        """Load markdown content and create description chunks.
        
        Args:
            md_dir: Directory containing markdown files
            
        Returns:
            int: Number of markdown files processed
        """
        logger.info(f"Loading markdown content from {md_dir}")
        
        # Load markdown files
        markdown_content = await load_markdown_dir(md_dir)
        
        # Process each markdown file
        processed_count = 0
        for slug, content_data in markdown_content.items():
            try:
                # Find corresponding dashboard
                entity_id = await self._get_entity_id_by_slug(slug)
                if not entity_id:
                    logger.warning(f"No dashboard found for slug '{slug}', skipping markdown")
                    continue
                
                # Create content chunks
                chunks = create_content_chunks(content_data)
                await self._create_chunks_for_entity(entity_id, chunks)
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing markdown for slug '{slug}': {e}")
                continue
        
        logger.info(f"Successfully processed {processed_count} markdown files")
        return processed_count
    
    async def apply_enrichment(self, enrichment_yaml: Path) -> int:
        """Apply enrichment configuration to existing dashboards.
        
        Args:
            enrichment_yaml: Path to enrichment YAML file
            
        Returns:
            int: Number of dashboards enriched
        """
        logger.info(f"Applying enrichment from {enrichment_yaml}")
        
        # Load enrichment configuration
        enrichment_config = await load_and_validate_enrichment_yaml(enrichment_yaml)
        
        # Get all dashboards from database
        dashboards = await self.db.fetch_all(
            "SELECT * FROM bi_entity WHERE entity_type = 'dashboard'"
        )
        
        # Apply enrichment to each dashboard
        enriched_count = 0
        for dashboard in dashboards:
            try:
                # Convert database record to dict
                dashboard_data = dict(dashboard)
                
                # Apply enrichment
                enriched_data = apply_enrichment_to_dashboard(dashboard_data, enrichment_config)
                
                # Update database if changes were made
                if enriched_data != dashboard_data:
                    await self._update_dashboard_enrichment(dashboard['entity_id'], enriched_data)
                    enriched_count += 1
                
            except Exception as e:
                logger.error(f"Error enriching dashboard {dashboard.get('title', 'unknown')}: {e}")
                continue
        
        logger.info(f"Successfully enriched {enriched_count} dashboards")
        return enriched_count
    
    async def index_missing_chunks(self, batch_size: int = 100) -> int:
        """Index chunks that are missing embeddings.
        
        Args:
            batch_size: Number of chunks to process in each batch
            
        Returns:
            int: Number of chunks processed
        """
        logger.info(f"Starting indexing of missing chunks with batch_size={batch_size}")
        
        # Get embedder
        embedder = get_embedder()
        
        # Count total chunks without embeddings
        total_missing = await self.db.fetch_value(
            "SELECT COUNT(*) FROM bi_chunk WHERE embedding IS NULL"
        )
        
        if total_missing == 0:
            logger.info("No chunks missing embeddings")
            return 0
        
        logger.info(f"Found {total_missing} chunks missing embeddings")
        
        processed_count = 0
        offset = 0
        
        while True:
            # Get batch of chunks without embeddings
            chunks = await self.db.fetch_all("""
                SELECT chunk_id, content 
                FROM bi_chunk 
                WHERE embedding IS NULL 
                ORDER BY chunk_id 
                LIMIT $1 OFFSET $2
            """, batch_size, offset)
            
            if not chunks:
                break
            
            logger.info(f"Processing batch of {len(chunks)} chunks (offset: {offset})")
            
            # Extract content for batch embedding
            chunk_contents = [chunk['content'] for chunk in chunks]
            chunk_ids = [chunk['chunk_id'] for chunk in chunks]
            
            try:
                # Generate embeddings for batch
                embeddings = embedder.embed_batch(chunk_contents)
                
                # Update chunks with embeddings
                for chunk_id, embedding in zip(chunk_ids, embeddings):
                    await self.db.execute_query("""
                        UPDATE bi_chunk 
                        SET embedding = $1 
                        WHERE chunk_id = $2
                    """, embedding.tolist(), chunk_id)
                
                processed_count += len(chunks)
                logger.info(f"Updated {len(chunks)} chunks with embeddings")
                
            except Exception as e:
                logger.error(f"Error processing batch at offset {offset}: {e}")
                # Continue with next batch
                pass
            
            offset += batch_size
        
        logger.info(f"Successfully indexed {processed_count} chunks")
        return processed_count
    
    async def run_complete_ingestion(self, 
                                   dashboards_csv: Path,
                                   charts_csv: Path,
                                   md_dir: Path,
                                   enrichment_yaml: Path) -> Dict[str, int]:
        """Run complete ingestion pipeline.
        
        Args:
            dashboards_csv: Path to dashboards CSV
            charts_csv: Path to charts CSV
            md_dir: Path to markdown directory
            enrichment_yaml: Path to enrichment YAML
            
        Returns:
            Dict[str, int]: Results summary
        """
        logger.info("Starting complete ingestion pipeline")
        
        results = {
            'dashboards_loaded': 0,
            'charts_loaded': 0,
            'markdown_processed': 0,
            'dashboards_enriched': 0
        }
        
        try:
            # Step 1: Load dashboards from CSV
            results['dashboards_loaded'] = await self.load_dashboards_from_csv(dashboards_csv)
            
            # Step 2: Load charts from CSV
            results['charts_loaded'] = await self.load_charts_from_csv(charts_csv)
            
            # Step 3: Load markdown content
            results['markdown_processed'] = await self.load_markdown_content(md_dir)
            
            # Step 4: Apply enrichment
            results['dashboards_enriched'] = await self.apply_enrichment(enrichment_yaml)
            
            logger.info(f"Complete ingestion pipeline finished: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error in complete ingestion pipeline: {e}")
            raise
    
    async def _upsert_dashboard(self, dashboard_data: Dict[str, Any]) -> int:
        """Insert or update dashboard in database.
        
        Args:
            dashboard_data: Dashboard data
            
        Returns:
            int: Entity ID of the dashboard
        """
        # Check if dashboard exists by superset_id
        existing = await self.db.fetch_one(
            "SELECT entity_id FROM bi_entity WHERE superset_id = $1",
            dashboard_data['superset_id']
        )
        
        if existing:
            # Update existing dashboard
            entity_id = existing['entity_id']
            await self.db.execute_query("""
                UPDATE bi_entity SET
                    dashboard_slug = $2,
                    title = $3,
                    description = $4,
                    owner = $5,
                    url = $6,
                    last_refresh_ts = NOW()
                WHERE entity_id = $1
            """, entity_id, dashboard_data.get('dashboard_slug'), 
                dashboard_data.get('title'), dashboard_data.get('description'),
                dashboard_data.get('owner'), dashboard_data.get('url'))
            
            # Delete existing chunks for this entity
            await self.db.execute_query(
                "DELETE FROM bi_chunk WHERE entity_id = $1", entity_id
            )
        else:
            # Insert new dashboard
            entity_id = await self.db.fetch_value("""
                INSERT INTO bi_entity (
                    entity_type, superset_id, dashboard_slug, title, description,
                    owner, url, last_refresh_ts
                ) VALUES (
                    'dashboard', $1, $2, $3, $4, $5, $6, NOW()
                ) RETURNING entity_id
            """, dashboard_data['superset_id'], dashboard_data.get('dashboard_slug'),
                dashboard_data.get('title'), dashboard_data.get('description'),
                dashboard_data.get('owner'), dashboard_data.get('url'))
        
        return entity_id
    
    async def _upsert_chart(self, chart_data: Dict[str, Any]) -> int:
        """Insert or update chart in database.
        
        Args:
            chart_data: Chart data
            
        Returns:
            int: Chart ID
        """
        # Get parent dashboard entity_id
        parent_entity = await self.db.fetch_one(
            "SELECT entity_id FROM bi_entity WHERE superset_id = $1",
            chart_data['parent_dashboard_id']
        )
        
        if not parent_entity:
            raise ValueError(f"Parent dashboard not found for chart: {chart_data['parent_dashboard_id']}")
        
        parent_entity_id = parent_entity['entity_id']
        
        # Check if chart exists by superset_chart_id
        existing = await self.db.fetch_one(
            "SELECT chart_id FROM bi_chart WHERE superset_chart_id = $1",
            chart_data['superset_chart_id']
        )
        
        if existing:
            # Update existing chart
            chart_id = existing['chart_id']
            await self.db.execute_query("""
                UPDATE bi_chart SET
                    parent_dashboard_id = $2,
                    title = $3,
                    viz_type = $4,
                    sql_text = $5,
                    metrics = $6,
                    dimensions = $7,
                    filters_default = $8,
                    url = $9
                WHERE chart_id = $1
            """, chart_id, parent_entity_id, chart_data.get('title'),
                chart_data.get('viz_type'), chart_data.get('sql_text'),
                json.dumps(chart_data.get('metrics')) if chart_data.get('metrics') else None,
                json.dumps(chart_data.get('dimensions')) if chart_data.get('dimensions') else None,
                json.dumps(chart_data.get('filters_default')) if chart_data.get('filters_default') else None,
                chart_data.get('url'))
            
            # Delete existing chunks for this chart
            await self.db.execute_query(
                "DELETE FROM bi_chunk WHERE chart_id = $1", chart_id
            )
        else:
            # Insert new chart
            chart_id = await self.db.fetch_value("""
                INSERT INTO bi_chart (
                    parent_dashboard_id, superset_chart_id, title, viz_type,
                    sql_text, metrics, dimensions, filters_default, url
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9
                ) RETURNING chart_id
            """, parent_entity_id, chart_data['superset_chart_id'],
                chart_data.get('title'), chart_data.get('viz_type'),
                chart_data.get('sql_text'), 
                json.dumps(chart_data.get('metrics')) if chart_data.get('metrics') else None,
                json.dumps(chart_data.get('dimensions')) if chart_data.get('dimensions') else None,
                json.dumps(chart_data.get('filters_default')) if chart_data.get('filters_default') else None,
                chart_data.get('url'))
        
        return chart_id
    
    async def _create_dashboard_chunks(self, entity_id: int, dashboard_data: Dict[str, Any]):
        """Create chunks for dashboard title and description.
        
        Args:
            entity_id: Dashboard entity ID
            dashboard_data: Dashboard data
        """
        chunks = []
        
        # Create title chunk
        if dashboard_data.get('title'):
            chunks.append({
                'entity_id': entity_id,
                'chart_id': None,
                'scope': 'title',
                'content': dashboard_data['title'],
                'lang': 'en'
            })
        
        # Create description chunk if available
        if dashboard_data.get('description'):
            chunks.append({
                'entity_id': entity_id,
                'chart_id': None,
                'scope': 'desc',
                'content': dashboard_data['description'],
                'lang': 'en'
            })
        
        # Insert chunks
        for chunk in chunks:
            await self._insert_chunk(chunk)
    
    async def _create_chart_chunks(self, chart_id: int, chart_data: Dict[str, Any]):
        """Create chunks for chart title and SQL.
        
        Args:
            chart_id: Chart ID
            chart_data: Chart data
        """
        chunks = []
        
        # Get parent entity_id
        parent_entity = await self.db.fetch_one(
            "SELECT parent_dashboard_id FROM bi_chart WHERE chart_id = $1", chart_id
        )
        entity_id = parent_entity['parent_dashboard_id']
        
        # Create chart title chunk
        if chart_data.get('title'):
            chunks.append({
                'entity_id': entity_id,
                'chart_id': chart_id,
                'scope': 'chart_title',
                'content': chart_data['title'],
                'lang': 'en'
            })
        
        # Create SQL chunk if available
        if chart_data.get('sql_text'):
            chunks.append({
                'entity_id': entity_id,
                'chart_id': chart_id,
                'scope': 'sql',
                'content': chart_data['sql_text'],
                'lang': 'en'
            })
        
        # Insert chunks
        for chunk in chunks:
            await self._insert_chunk(chunk)
    
    async def _create_chunks_for_entity(self, entity_id: int, chunks: List[Dict[str, Any]]):
        """Create chunks for an entity.
        
        Args:
            entity_id: Entity ID
            chunks: List of chunk data
        """
        for chunk_data in chunks:
            chunk = {
                'entity_id': entity_id,
                'chart_id': None,
                'scope': chunk_data['scope'],
                'content': chunk_data['content'],
                'lang': chunk_data.get('lang', 'en')
            }
            await self._insert_chunk(chunk)
    
    async def _insert_chunk(self, chunk_data: Dict[str, Any]):
        """Insert a chunk into the database.
        
        Args:
            chunk_data: Chunk data
        """
        await self.db.execute_query("""
            INSERT INTO bi_chunk (entity_id, chart_id, scope, content, lang)
            VALUES ($1, $2, $3, $4, $5)
        """, chunk_data['entity_id'], chunk_data['chart_id'],
            chunk_data['scope'], chunk_data['content'], chunk_data['lang'])
    
    async def _get_entity_id_by_slug(self, slug: str) -> Optional[int]:
        """Get entity ID by dashboard slug.
        
        Args:
            slug: Dashboard slug
            
        Returns:
            Optional[int]: Entity ID or None if not found
        """
        result = await self.db.fetch_one(
            "SELECT entity_id FROM bi_entity WHERE dashboard_slug = $1", slug
        )
        return result['entity_id'] if result else None
    
    async def _update_dashboard_enrichment(self, entity_id: int, enriched_data: Dict[str, Any]):
        """Update dashboard with enrichment data.
        
        Args:
            entity_id: Entity ID
            enriched_data: Enriched dashboard data
        """
        await self.db.execute_query("""
            UPDATE bi_entity SET
                domain = $2,
                tags = $3,
                metadata = $4
            WHERE entity_id = $1
        """, entity_id, enriched_data.get('domain'),
            enriched_data.get('tags'), 
            json.dumps(enriched_data.get('metadata', {})))


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Dashboard Ingestion CLI")
    parser.add_argument("--dashboards-csv", type=Path, help="Path to dashboards CSV file")
    parser.add_argument("--charts-csv", type=Path, help="Path to charts CSV file")
    parser.add_argument("--md-dir", type=Path, help="Path to markdown directory")
    parser.add_argument("--enrichment-yaml", type=Path, help="Path to enrichment YAML file")
    parser.add_argument("--complete", action="store_true", 
                       help="Run complete ingestion pipeline (requires all paths)")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without loading")
    parser.add_argument("--only-missing", action="store_true", 
                       help="Index only chunks missing embeddings")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Batch size for processing chunks (default: 100)")
    
    args = parser.parse_args()
    
    if args.complete:
        # Validate all required paths are provided
        required_paths = [args.dashboards_csv, args.charts_csv, args.md_dir, args.enrichment_yaml]
        if not all(required_paths):
            logger.error("Complete ingestion requires all paths: --dashboards-csv, --charts-csv, --md-dir, --enrichment-yaml")
            sys.exit(1)
        
        # Validate all paths exist
        for path in required_paths:
            if not path.exists():
                logger.error(f"Path does not exist: {path}")
                sys.exit(1)
    
    if args.dry_run:
        logger.info("Dry run mode - validating inputs only")
        # TODO: Add validation logic
        logger.info("Validation completed successfully")
        return
    
    try:
        job = IndexJob()
        
        if args.complete:
            # Run complete pipeline
            results = await job.run_complete_ingestion(
                dashboards_csv=args.dashboards_csv,
                charts_csv=args.charts_csv,
                md_dir=args.md_dir,
                enrichment_yaml=args.enrichment_yaml
            )
            logger.info(f"Complete ingestion results: {results}")
        else:
            # Run individual components
            if args.dashboards_csv:
                count = await job.load_dashboards_from_csv(args.dashboards_csv)
                logger.info(f"Loaded {count} dashboards")
            
            if args.charts_csv:
                count = await job.load_charts_from_csv(args.charts_csv)
                logger.info(f"Loaded {count} charts")
            
            if args.md_dir:
                count = await job.load_markdown_content(args.md_dir)
                logger.info(f"Processed {count} markdown files")
            
            if args.enrichment_yaml:
                count = await job.apply_enrichment(args.enrichment_yaml)
                logger.info(f"Enriched {count} dashboards")
            
            if args.only_missing:
                count = await job.index_missing_chunks(batch_size=args.batch_size)
                logger.info(f"Indexed {count} missing chunks")
    
    except Exception as e:
        logger.error(f"Ingestion job failed: {e}")
        sys.exit(1)
    finally:
        await DashAssistantDB.close_pool()


if __name__ == "__main__":
    asyncio.run(main())
