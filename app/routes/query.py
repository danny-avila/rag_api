import traceback
from fastapi import APIRouter, Request, HTTPException
from app.models import QueryRequestBody, QueryMultipleBody
from app.config import vector_store, logger
from app.store.vector import AsyncPgVector
from app.store.vector import run_in_executor

router = APIRouter()

@router.post("/query")
async def query_embeddings_by_file_id(body: QueryRequestBody, request: Request):
    if not hasattr(request.state, "user"):
        user_authorized = body.entity_id if body.entity_id else "public"
    else:
        user_authorized = body.entity_id if body.entity_id else request.state.user.get("id")
    authorized_documents = []
    try:
        embedding = vector_store.embedding_function.embed_query(body.query)
        if isinstance(vector_store, AsyncPgVector):
            documents = await run_in_executor(None, vector_store.similarity_search_with_score_by_vector, embedding, body.k, {"file_id": body.file_id})
        else:
            documents = vector_store.similarity_search_with_score_by_vector(embedding, k=body.k, filter={"file_id": body.file_id})
        if not documents:
            return authorized_documents
        document, score = documents[0]
        doc_metadata = document.metadata
        doc_user_id = doc_metadata.get("user_id")
        if doc_user_id is None or doc_user_id == user_authorized:
            authorized_documents = documents
        else:
            if body.entity_id and hasattr(request.state, "user"):
                user_authorized = request.state.user.get("id")
                if doc_user_id == user_authorized:
                    authorized_documents = documents
                else:
                    if body.entity_id == doc_user_id:
                        logger.warning(f"Entity ID {body.entity_id} matches document user_id but user {user_authorized} is not authorized")
                    else:
                        logger.warning(f"Access denied for both entity ID {body.entity_id} and user {user_authorized} to document with user_id {doc_user_id}")
            else:
                logger.warning(f"Unauthorized access attempt by user {user_authorized} to a document with user_id {doc_user_id}")
        return authorized_documents
    except Exception as e:
        logger.error("Error in query embeddings | File ID: %s | Query: %s | Error: %s | Traceback: %s", body.file_id, body.query, str(e), traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query_multiple")
async def query_embeddings_by_file_ids(body: QueryMultipleBody):
    try:
        embedding = vector_store.embedding_function.embed_query(body.query)
        if isinstance(vector_store, AsyncPgVector):
            documents = await run_in_executor(None, vector_store.similarity_search_with_score_by_vector, embedding, body.k, {"file_id": {"$in": body.file_ids}})
        else:
            documents = vector_store.similarity_search_with_score_by_vector(embedding, k=body.k, filter={"file_id": {"$in": body.file_ids}})
        if not documents:
            raise HTTPException(status_code=404, detail="No documents found for the given query")
        return documents
    except Exception as e:
        logger.error("Error in query multiple embeddings | File IDs: %s | Query: %s | Error: %s | Traceback: %s", body.file_ids, body.query, str(e), traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))