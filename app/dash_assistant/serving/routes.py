# app/dash_assistant/serving/routes.py
"""FastAPI routes for dash assistant."""
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Literal
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.config import logger
from app.dash_assistant.db import DashAssistantDB
from app.dash_assistant.ingestion.index_jobs import IndexJob
from app.dash_assistant.config import get_config
from .retriever import DashRetriever
from .answer_builder import build_answer


# Request/Response Models
class IngestRequest(BaseModel):
    """Request model for ingestion endpoint."""
    dashboards_csv: str = Field(..., description="Path to dashboards CSV file")
    charts_csv: str = Field(..., description="Path to charts CSV file")
    md_dir: str = Field(..., description="Path to markdown directory")
    enrichment_yaml: str = Field(..., description="Path to enrichment YAML file")
    run_embeddings: bool = Field(False, description="Whether to run embeddings indexing")


class IngestResponse(BaseModel):
    """Response model for ingestion endpoint."""
    status: str
    results: Dict[str, int]
    message: str


def _get_default_topk() -> int:
    """Get default top_k from config."""
    return get_config().default_topk


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    q: str = Field(..., description="Search query")
    top_k: int = Field(default_factory=_get_default_topk, description="Number of results to return", ge=1, le=50)
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional filters")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    results: List[Dict[str, Any]]
    debug: Dict[str, Any]


class FeedbackRequest(BaseModel):
    """Request model for feedback endpoint."""
    qid: int = Field(..., description="Query ID from query_log", ge=1)
    entity_id: int = Field(..., description="Entity ID that was chosen", ge=1)
    chart_id: Optional[int] = Field(None, description="Chart ID if specific chart was chosen", ge=1)
    feedback: Literal["up", "down"] = Field(..., description="User feedback: up (👍) or down (👎)")


class FeedbackResponse(BaseModel):
    """Response model for feedback endpoint."""
    status: str
    message: str


# Router
router = APIRouter(prefix="/dash", tags=["dash-assistant"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_data(request: IngestRequest):
    """Ingest dashboard data from CSV files and markdown.
    
    This endpoint runs the complete ingestion pipeline:
    1. Load dashboards from CSV
    2. Load charts from CSV
    3. Load markdown content
    4. Apply enrichment rules
    5. Optionally run embeddings indexing
    """
    logger.info(f"Starting ingestion with request: {request}")
    
    # Check database health
    if not await DashAssistantDB.health_check():
        logger.error("Database health check failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is not available"
        )
    
    # Validate file paths exist
    paths_to_check = [
        Path(request.dashboards_csv),
        Path(request.charts_csv),
        Path(request.md_dir),
        Path(request.enrichment_yaml)
    ]
    
    for path in paths_to_check:
        if not path.exists():
            logger.error(f"Path does not exist: {path}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Path does not exist: {path}"
            )
    
    try:
        # Initialize ingestion job
        job = IndexJob()
        
        # Run ingestion steps
        logger.info("Starting dashboard ingestion pipeline")
        
        # Step 1: Load dashboards
        dashboards_loaded = await job.load_dashboards_from_csv(Path(request.dashboards_csv))
        logger.info(f"Loaded {dashboards_loaded} dashboards")
        
        # Step 2: Load charts
        charts_loaded = await job.load_charts_from_csv(Path(request.charts_csv))
        logger.info(f"Loaded {charts_loaded} charts")
        
        # Step 3: Load markdown content
        markdown_processed = await job.load_markdown_content(Path(request.md_dir))
        logger.info(f"Processed {markdown_processed} markdown files")
        
        # Step 4: Apply enrichment
        dashboards_enriched = await job.apply_enrichment(Path(request.enrichment_yaml))
        logger.info(f"Enriched {dashboards_enriched} dashboards")
        
        # Prepare results
        results = {
            "dashboards_loaded": dashboards_loaded,
            "charts_loaded": charts_loaded,
            "markdown_processed": markdown_processed,
            "dashboards_enriched": dashboards_enriched
        }
        
        # Step 5: Optionally run embeddings indexing
        if request.run_embeddings:
            logger.info("Running embeddings indexing")
            chunks_indexed = await job.index_missing_chunks(batch_size=100)
            results["chunks_indexed"] = chunks_indexed
            logger.info(f"Indexed {chunks_indexed} chunks")
        
        total_processed = sum(results.values())
        message = f"Successfully processed {total_processed} items"
        
        logger.info(f"Ingestion completed: {results}")
        
        return IngestResponse(
            status="success",
            results=results,
            message=message
        )
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )


@router.post("/query", response_model=QueryResponse)
async def query_dashboards(request: QueryRequest):
    """Query dashboards using multi-signal search with RRF.
    
    This endpoint:
    1. Runs parallel searches (FTS, Vector, Trigram)
    2. Combines results using Reciprocal Rank Fusion (RRF)
    3. Builds structured answers with explanations
    4. Returns results with debug information
    """
    logger.info(f"Processing query: '{request.q}' with top_k={request.top_k}")
    
    # Check database health
    if not await DashAssistantDB.health_check():
        logger.error("Database health check failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is not available"
        )
    
    try:
        # Initialize retriever
        retriever = DashRetriever()
        
        # Search for candidates
        candidates = await retriever.search(
            query=request.q,
            top_k=request.top_k,
            filters=request.filters
        )
        
        logger.debug(f"Found {len(candidates)} candidates")
        
        # Build structured answer
        answer = build_answer(request.q, candidates)
        
        # Prepare debug information
        debug_info = {
            "query": request.q,
            "top_k": request.top_k,
            "filters": request.filters,
            "candidates_found": len(candidates),
            "processing_time_ms": None  # Could add timing if needed
        }
        
        # Add individual signal scores to debug if available
        if candidates:
            debug_info["signal_breakdown"] = {}
            for candidate in candidates[:3]:  # Show top 3 for debugging
                entity_id = candidate.get('entity_id')
                if entity_id:
                    debug_info["signal_breakdown"][str(entity_id)] = {
                        "title": candidate.get('title', ''),
                        "rrf_score": candidate.get('score', 0),
                        "signal_sources": candidate.get('signal_sources', []),
                        "individual_scores": candidate.get('individual_scores', {})
                    }
        
        logger.info(f"Query completed: {len(answer['results'])} results returned")
        
        return QueryResponse(
            results=answer['results'],
            debug=debug_info
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint for dash assistant."""
    try:
        # Check database connectivity
        db_healthy = await DashAssistantDB.health_check()
        
        if db_healthy:
            return {
                "status": "healthy",
                "database": "connected",
                "service": "dash-assistant"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database connection failed"
            )
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )


@router.post("/feedback", response_model=FeedbackResponse)
async def record_feedback(request: FeedbackRequest):
    """Record user feedback for search results.
    
    This endpoint records user feedback (👍/👎) for search results
    and updates the query_log table with the feedback and chosen entities.
    """
    logger.info(f"Recording feedback: qid={request.qid}, entity_id={request.entity_id}, feedback={request.feedback}")
    
    # Check database health
    if not await DashAssistantDB.health_check():
        logger.error("Database health check failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is not available"
        )
    
    try:
        # Update query_log with feedback and chosen entities
        update_query = """
            UPDATE query_log 
            SET feedback = $1, chosen_entity = $2, chosen_chart = $3
            WHERE qid = $4
        """
        
        await DashAssistantDB.execute_query(
            update_query,
            request.feedback,
            request.entity_id,
            request.chart_id,
            request.qid
        )
        
        logger.info(f"Feedback recorded successfully for qid={request.qid}")
        
        return FeedbackResponse(
            status="success",
            message=f"Feedback '{request.feedback}' recorded for query {request.qid}"
        )
        
    except Exception as e:
        logger.error(f"Feedback recording failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feedback recording failed: {str(e)}"
        )


@router.get("/stats")
async def get_stats():
    """Get statistics about the dash assistant data."""
    try:
        # Check database health first
        if not await DashAssistantDB.health_check():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database is not available"
            )
        
        # Get counts
        dashboards_count = await DashAssistantDB.fetch_value(
            "SELECT COUNT(*) FROM bi_entity WHERE entity_type = 'dashboard'"
        )
        
        charts_count = await DashAssistantDB.fetch_value(
            "SELECT COUNT(*) FROM bi_chart"
        )
        
        chunks_count = await DashAssistantDB.fetch_value(
            "SELECT COUNT(*) FROM bi_chunk"
        )
        
        chunks_with_embeddings = await DashAssistantDB.fetch_value(
            "SELECT COUNT(*) FROM bi_chunk WHERE embedding IS NOT NULL"
        )
        
        # Get top domains
        top_domains = await DashAssistantDB.fetch_all("""
            SELECT domain, COUNT(*) as count
            FROM bi_entity 
            WHERE domain IS NOT NULL 
            GROUP BY domain 
            ORDER BY count DESC 
            LIMIT 5
        """)
        
        stats = {
            "dashboards": dashboards_count or 0,
            "charts": charts_count or 0,
            "chunks": chunks_count or 0,
            "chunks_with_embeddings": chunks_with_embeddings or 0,
            "embedding_coverage": (
                (chunks_with_embeddings / chunks_count * 100) 
                if chunks_count > 0 else 0
            ),
            "top_domains": [
                {"domain": row["domain"], "count": row["count"]} 
                for row in top_domains
            ]
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stats retrieval failed: {str(e)}"
        )
