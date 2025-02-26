import traceback
from fastapi import APIRouter, Query, HTTPException, status
from app.config import logger, vector_store
from app.models import DocumentResponse
from app.store.vector import AsyncPgVector
from app.parsers import process_documents

router = APIRouter()

@router.get("/ids")
async def get_all_ids():
    try:
        if isinstance(vector_store, AsyncPgVector):
            ids = await vector_store.get_all_ids()
        else:
            ids = vector_store.get_all_ids()
        return list(set(ids))
    except Exception as e:
        logger.error("Failed to get all IDs | Error: %s | Traceback: %s", str(e), traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents", response_model=list[DocumentResponse])
async def get_documents_by_ids(ids: list[str] = Query(...)):
    try:
        if isinstance(vector_store, AsyncPgVector):
            existing_ids = await vector_store.get_all_ids()
            documents = await vector_store.get_documents_by_ids(ids)
        else:
            existing_ids = vector_store.get_all_ids()
            documents = vector_store.get_documents_by_ids(ids)
        if not all(id in existing_ids for id in ids):
            raise HTTPException(status_code=404, detail="One or more IDs not found")
        if not documents:
            raise HTTPException(status_code=404, detail="No documents found for the given IDs")
        return documents
    except HTTPException as http_exc:
        logger.error("HTTP Exception in get_documents_by_ids | Status: %d | Detail: %s", http_exc.status_code, http_exc.detail)
        raise http_exc
    except Exception as e:
        logger.error("Error getting documents by IDs | IDs: %s | Error: %s | Traceback: %s", ids, str(e), traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/documents")
async def delete_documents(document_ids: list[str]):
    try:
        if isinstance(vector_store, AsyncPgVector):
            existing_ids = await vector_store.get_all_ids()
            await vector_store.delete(ids=document_ids)
        else:
            existing_ids = vector_store.get_all_ids()
            vector_store.delete(ids=document_ids)
        if not all(id in existing_ids for id in document_ids):
            raise HTTPException(status_code=404, detail="One or more IDs not found")
        file_count = len(document_ids)
        return {"message": f"Documents for {file_count} file{'s' if file_count > 1 else ''} deleted successfully"}
    except Exception as e:
        logger.error("Failed to delete documents | IDs: %s | Error: %s | Traceback: %s", document_ids, str(e), traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents/{id}/context")
async def load_document_context(id: str):
    ids = [id]
    try:
        if isinstance(vector_store, AsyncPgVector):
            existing_ids = await vector_store.get_all_ids()
            documents = await vector_store.get_documents_by_ids(ids)
        else:
            existing_ids = vector_store.get_all_ids()
            documents = vector_store.get_documents_by_ids(ids)
        if not all(id in existing_ids for id in ids):
            raise HTTPException(status_code=404, detail="The specified file_id was not found")
        if not documents:
            raise HTTPException(status_code=404, detail="No document found for the given ID")
        return process_documents(documents)
    except Exception as e:
        logger.error("Error loading document context | Document ID: %s | Error: %s | Traceback: %s", id, str(e), traceback.format_exc())
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))