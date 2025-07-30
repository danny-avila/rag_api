from fastapi import (
    APIRouter,
    Request,
    UploadFile,
    HTTPException,
    File,
    Form,
    Body,
    Query,
)
from typing import List
import os
import aiofiles
import aiofiles.os
from langchain_core.runnables import run_in_executor
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.services.vector_store.factory import VectorStoreManager
from app.services.kb_manager import KBManager
from app.config import (
    CONNECTION_STRING,
    embeddings,
    logger,
    RAG_UPLOAD_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from app.models import QueryRequestBody, DocumentResponse
from app.utils.document_loader import get_loader, cleanup_temp_encoding_file

router = APIRouter(prefix="/v2/knowledge-bases", tags=["V2 KB Operations"])


async def get_kb_vector_store(kb_id: str):
    """Get vector store for a specific KB"""
    if not KBManager.validate_kb_id(kb_id):
        raise HTTPException(status_code=400, detail=f"Invalid KB ID format: {kb_id}")

    return VectorStoreManager.get_vector_store(
        kb_id=kb_id,
        connection_string=CONNECTION_STRING,
        embeddings=embeddings,
        mode="async",
    )


def calculate_file_size(file_path: str) -> int:
    """Calculate file size in bytes"""
    try:
        return os.path.getsize(file_path)
    except Exception:
        return 0


def get_cached_query_embedding(query: str):
    """Get cached query embedding - imported from document_routes logic"""
    # Import here to avoid circular imports
    from app.routes.document_routes import (
        get_cached_query_embedding as _get_cached_query_embedding,
    )

    return _get_cached_query_embedding(query)


def generate_digest(page_content: str):
    """Generate digest - imported from document_routes logic"""
    from app.routes.document_routes import generate_digest as _generate_digest

    return _generate_digest(page_content)


def clean_text(text: str):
    """Clean text - imported from document_routes logic"""
    from app.utils.document_loader import clean_text as _clean_text

    return _clean_text(text)


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "Knowledge Base routes running properly"}


@router.post("/{kb_id}/embed")
async def embed_file_to_kb(
    kb_id: str,
    request: Request,
    file_id: str = Form(
        ...
    ),  # file ID is optional for query, but required as a response for citation.
    file: UploadFile = File(...),
    entity_id: str = Form(None),
):
    """Embed a file into a specific knowledge base"""
    logger.info(f"Embedding file {file.filename} to KB {kb_id} with ID {file_id}")
    try:
        vector_store = await get_kb_vector_store(kb_id)
    except Exception as e:
        logger.error(f"Failed to get vector store for KB {kb_id}: {str(e)}")
        raise HTTPException(
            status_code=404, detail=f"Knowledge base not found: {kb_id}"
        )

    if not hasattr(request.state, "user"):
        user_id = entity_id if entity_id else "public"
    else:
        user_id = entity_id if entity_id else request.state.user.get("id")

    temp_base_path = os.path.join(RAG_UPLOAD_DIR, user_id)
    os.makedirs(temp_base_path, exist_ok=True)
    temp_file_path = os.path.join(temp_base_path, file.filename)

    try:
        # Save uploaded file
        async with aiofiles.open(temp_file_path, "wb") as temp_file:
            chunk_size = 64 * 1024
            while content := await file.read(chunk_size):
                await temp_file.write(content)

        # Calculate file size
        file_size_bytes = calculate_file_size(temp_file_path)

        # Process file
        loader, known_type, file_ext = get_loader(
            file.filename, file.content_type, temp_file_path
        )
        data = await run_in_executor(request.app.state.thread_pool, loader.load)
        cleanup_temp_encoding_file(loader)

        # Store in KB-specific vector store
        result = await store_data_in_vector_db_kb(
            vector_store=vector_store,
            data=data,
            file_id=file_id,
            user_id=user_id,
            clean_content=file_ext == "pdf",
            executor=request.app.state.thread_pool,
        )

        if not result or "error" in result:
            raise HTTPException(
                status_code=500, detail="Failed to process/store the file data."
            )

        logger.info(f"File embedded successfully in KB {kb_id}: {file_id}")

        return {
            "kb_id": kb_id,
            "file_id": file_id,
            "filename": file.filename,
            "known_type": known_type,
            "file_size_bytes": file_size_bytes,
            "chunk_count": len(result.get("ids", [])),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error embedding file to KB {kb_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            await aiofiles.os.remove(temp_file_path)
        except Exception as e:
            logger.error(f"Failed to cleanup temp file: {str(e)}")


@router.get("/{kb_id}/documents", response_model=List[DocumentResponse])
async def get_kb_documents(kb_id: str, request: Request, ids: List[str] = Query(...)):
    """Get documents from a specific knowledge base"""
    logger.info(f"Retrieving documents from KB {kb_id} for IDs: {ids}")
    try:
        vector_store = await get_kb_vector_store(kb_id)

        if hasattr(vector_store, "get_documents_by_ids"):
            documents = await vector_store.get_documents_by_ids(
                ids, executor=request.app.state.thread_pool
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Vector store doesn't support document retrieval",
            )

        if not documents:
            raise HTTPException(
                status_code=404, detail="No documents found for the given IDs"
            )

        # Add kb_id to each document's metadata
        for doc in documents:
            if hasattr(doc, "metadata"):
                doc.metadata["kb_id"] = kb_id

        logger.info(f"Retrieved {len(documents)} documents from KB {kb_id}")
        return documents

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving documents from KB {kb_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{kb_id}/documents")
async def delete_kb_documents(
    kb_id: str, request: Request, document_ids: List[str] = Body(...)
):
    """Delete documents from a specific knowledge base"""
    try:
        vector_store = await get_kb_vector_store(kb_id)

        if hasattr(vector_store, "delete"):
            await vector_store.delete(
                ids=document_ids, executor=request.app.state.thread_pool
            )
        else:
            raise HTTPException(
                status_code=500, detail="Vector store doesn't support document deletion"
            )

        logger.info(f"Deleted {len(document_ids)} documents from KB {kb_id}")

        return {
            "kb_id": kb_id,
            "message": f"Documents for {len(document_ids)} file{'s' if len(document_ids) > 1 else ''} deleted successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting documents from KB {kb_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{kb_id}/query")
async def query_kb(kb_id: str, body: QueryRequestBody, request: Request):
    """Query a specific knowledge base"""
    logger.info(f"Querying KB {kb_id} with query: {body.query}")
    try:
        vector_store = await get_kb_vector_store(kb_id)

        embedding = get_cached_query_embedding(body.query)

        if hasattr(vector_store, "asimilarity_search_with_score_by_vector"):
            documents = await vector_store.asimilarity_search_with_score_by_vector(
                embedding,
                k=body.k,
                filter={"file_id": body.file_id} if body.file_id else None,
                executor=request.app.state.thread_pool,
            )
        else:
            raise HTTPException(
                status_code=500, detail="Vector store doesn't support similarity search"
            )

        # Add kb_id to results
        results = []
        for doc, score in documents:
            result = {
                "content": doc.page_content,
                "metadata": {**doc.metadata, "kb_id": kb_id},
                "score": float(score),
            }
            results.append(result)

        logger.info(f"Query executed on KB {kb_id}, found {len(results)} results")
        return {"results": results, "kb_id": kb_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying KB {kb_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Cross-KB Query Implementation
@router.post("/query-multiple")
async def query_multiple_kbs(
    request: Request,
    kb_ids: List[str] = Body(...),
    query: str = Body(...),
    k: int = Body(5),
    max_kbs: int = 5,
):
    """Query across multiple knowledge bases"""
    if len(kb_ids) > max_kbs:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {max_kbs} knowledge bases allowed per query",
        )

    embedding = get_cached_query_embedding(query)
    all_results = []

    for kb_id in kb_ids:
        try:
            vector_store = await get_kb_vector_store(kb_id)

            documents = await vector_store.asimilarity_search_with_score_by_vector(
                embedding,
                k=k,
                executor=request.app.state.thread_pool,
            )

            for doc, score in documents:
                result = {
                    "content": doc.page_content,
                    "metadata": {**doc.metadata, "kb_id": kb_id},
                    "score": float(score),
                }
                all_results.append(result)

        except Exception as e:
            logger.warning(f"Failed to query KB {kb_id}: {str(e)}")
            continue

    # Sort by score (descending) and limit to k results
    all_results.sort(key=lambda x: x["score"], reverse=True)
    final_results = all_results[:k]

    logger.info(
        f"Cross-KB query executed across {len(kb_ids)} KBs, returning {len(final_results)} results"
    )

    return {
        "results": final_results,
        "queried_kbs": kb_ids,
        "total_results": len(final_results),
    }


# Helper function for KB-specific embedding storage
async def store_data_in_vector_db_kb(
    vector_store,
    data,
    file_id: str,
    user_id: str = "",
    clean_content: bool = False,
    executor=None,
) -> dict:
    """Store data in a specific KB's vector store"""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    documents = text_splitter.split_documents(data)

    if clean_content:
        for doc in documents:
            doc.page_content = clean_text(doc.page_content)

    docs = [
        Document(
            page_content=doc.page_content,
            metadata={
                "file_id": file_id,
                "user_id": user_id,
                "digest": generate_digest(doc.page_content),
                **(doc.metadata or {}),
            },
        )
        for doc in documents
    ]

    try:
        if hasattr(vector_store, "aadd_documents"):
            ids = await vector_store.aadd_documents(
                docs, ids=[file_id] * len(documents), executor=executor
            )
        else:
            ids = vector_store.add_documents(docs, ids=[file_id] * len(documents))

        return {"message": "Documents added successfully", "ids": ids}

    except Exception as e:
        logger.error(f"Failed to store data in KB vector store: {str(e)}")
        return {"message": "An error occurred while adding documents.", "error": str(e)}
