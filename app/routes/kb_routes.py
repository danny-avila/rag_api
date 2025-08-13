from fastapi import APIRouter, HTTPException, Query
from typing import List
from app.services.kb_manager import KBManager
from app.config import logger

router = APIRouter(prefix="/knowledge-bases", tags=["Knowledge Bases"])


# Create a new Knowledge Base.
@router.post("/{kb_id}")
async def create_knowledge_base(kb_id: str):
    """Create a new knowledge base"""
    print("Creating a new knowledge base with id ", kb_id)
    try:
        result = await KBManager.create_kb(kb_id)
        logger.info(f"KB creation requested: {kb_id}")
        return result
    except ValueError as e:
        logger.error(f"Invalid KB ID format: {kb_id}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create KB {kb_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Delete a Knowledge Base.
@router.delete("/{kb_id}")
async def delete_knowledge_base(kb_id: str):
    """Delete a knowledge base and all its data"""
    logger.info(f"Deleting KB: {kb_id}")
    try:
        result = await KBManager.delete_kb(kb_id)
        logger.info(f"KB deletion requested: {kb_id}")
        return result
    except Exception as e:
        logger.error(f"Failed to delete KB {kb_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Get basic information about the Knowledge base.
@router.get("")
async def get_knowledge_bases_info(kb_ids: List[str] = Query(...)):
    """Get information about multiple knowledge bases"""
    try:
        result = await KBManager.get_kb_info(kb_ids)
        logger.info(f"KB info requested for: {kb_ids}")
        return result
    except Exception as e:
        logger.error(f"Failed to get KB info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
