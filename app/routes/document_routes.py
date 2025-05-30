# app/routes/document_routes.py
import os, psutil
import hashlib
import traceback
import aiofiles
import aiofiles.os
import asyncio
from shutil import copyfileobj
from typing import List, Iterable, Optional
from fastapi import (
    APIRouter,
    Request,
    UploadFile,
    HTTPException,
    File,
    Form,
    Body,
    Query,
    status,
)
from langchain_core.documents import Document
from langchain_core.runnables import run_in_executor
from langchain_text_splitters import RecursiveCharacterTextSplitter
from functools import lru_cache

from app.config import (
    logger,
    vector_store,
    RAG_UPLOAD_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    file_storage_service,
)
from app.constants import ERROR_MESSAGES
from app.models import (
    StoreDocument,
    QueryRequestBody,
    DocumentResponse,
    QueryMultipleBody,
)
from app.services.vector_store.async_pg_vector import AsyncPgVector
from app.utils.document_loader import get_loader, clean_text, process_documents
from app.utils.health import is_health_ok

router = APIRouter()


@router.get("/ids")
async def get_all_ids():
    try:
        if isinstance(vector_store, AsyncPgVector):
            ids = await vector_store.get_all_ids()
        else:
            ids = vector_store.get_all_ids()

        return list(set(ids))
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in get_all_ids | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Failed to get all IDs | Error: %s | Traceback: %s",
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    try:
        if await is_health_ok():
            return {"status": "UP"}
        else:
            logger.error("Health check failed")
            return {"status": "DOWN"}, 503
    except Exception as e:
        logger.error(
            "Error during health check | Error: %s | Traceback: %s",
            str(e),
            traceback.format_exc(),
        )
        return {"status": "DOWN", "error": str(e)}, 503


@router.get("/documents", response_model=list[DocumentResponse])
async def get_documents_by_ids(ids: list[str] = Query(...)):
    try:
        if isinstance(vector_store, AsyncPgVector):
            existing_ids = await vector_store.get_filtered_ids(ids)
            documents = await vector_store.get_documents_by_ids(ids)
        else:
            existing_ids = vector_store.get_filtered_ids(ids)
            documents = vector_store.get_documents_by_ids(ids)

        # Ensure all requested ids exist
        if not all(id in existing_ids for id in ids):
            raise HTTPException(
                status_code=404, detail="One or more IDs not found"
            )

        # Ensure documents list is not empty
        if not documents:
            raise HTTPException(
                status_code=404, detail="No documents found for the given IDs"
            )

        return documents
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in get_documents_by_ids | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error getting documents by IDs | IDs: %s | Error: %s | Traceback: %s",
            ids,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents")
async def delete_documents(document_ids: List[str] = Body(...)):
    try:
        # Get documents first to extract storage metadata
        if isinstance(vector_store, AsyncPgVector):
            existing_ids = await vector_store.get_filtered_ids(document_ids)
            documents = await vector_store.get_documents_by_ids(document_ids)
        else:
            existing_ids = vector_store.get_filtered_ids(document_ids)
            documents = vector_store.get_documents_by_ids(document_ids)

        if not all(id in existing_ids for id in document_ids):
            raise HTTPException(
                status_code=404, detail="One or more IDs not found"
            )

        # Delete stored files if file_storage_service is available
        if file_storage_service and documents:
            storage_keys = set()  # Use set to avoid duplicate deletions
            for doc in documents:
                # Check for both local and S3 storage keys
                storage_key = doc.metadata.get(
                    "storage_key"
                ) or doc.metadata.get("s3_key")
                if storage_key:
                    storage_keys.add(storage_key)
            # Delete files from storage
            for storage_key in storage_keys:
                try:
                    deleted = await file_storage_service.delete_file(
                        storage_key
                    )
                    if deleted:
                        logger.info(f"Deleted stored file: {storage_key}")
                    else:
                        logger.warning(
                            f"Failed to delete stored file: {storage_key}"
                        )
                except Exception as e:
                    logger.error(
                        f"Error deleting stored file {storage_key}: {e}"
                    )

        # Delete vector embeddings
        if isinstance(vector_store, AsyncPgVector):
            await vector_store.delete(ids=document_ids)
        else:
            vector_store.delete(ids=document_ids)

        file_count = len(document_ids)
        return {
            "message": f"Documents for {file_count} file{'s' if file_count > 1 else ''} deleted successfully"
        }
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in delete_documents | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Failed to delete documents | IDs: %s | Error: %s | Traceback: %s",
            document_ids,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


# Cache the embedding function with LRU cache
@lru_cache(maxsize=128)
def get_cached_query_embedding(query: str):
    return vector_store.embedding_function.embed_query(query)


@router.post("/query")
async def query_embeddings_by_file_id(
    body: QueryRequestBody,
    request: Request,
):
    if not hasattr(request.state, "user"):
        user_authorized = body.entity_id if body.entity_id else "public"
    else:
        user_authorized = (
            body.entity_id if body.entity_id else request.state.user.get("id")
        )

    authorized_documents = []

    try:
        embedding = get_cached_query_embedding(body.query)

        if isinstance(vector_store, AsyncPgVector):
            documents = await run_in_executor(
                None,
                vector_store.similarity_search_with_score_by_vector,
                embedding,
                k=body.k,
                filter={"file_id": body.file_id},
            )
        else:
            documents = vector_store.similarity_search_with_score_by_vector(
                embedding, k=body.k, filter={"file_id": body.file_id}
            )

        if not documents:
            return authorized_documents

        document, score = documents[0]
        doc_metadata = document.metadata
        doc_user_id = doc_metadata.get("user_id")

        if doc_user_id is None or doc_user_id == user_authorized:
            authorized_documents = documents
        else:
            # If using entity_id and access denied, try again with user's actual ID
            if body.entity_id and hasattr(request.state, "user"):
                user_authorized = request.state.user.get("id")
                if doc_user_id == user_authorized:
                    authorized_documents = documents
                else:
                    if body.entity_id == doc_user_id:
                        logger.warning(
                            f"Entity ID {body.entity_id} matches document "
                            f"user_id but user {user_authorized} is not authorized"
                        )
                    else:
                        logger.warning(
                            f"Access denied for both entity ID {body.entity_id} "
                            f"and user {user_authorized} to document with user_id {doc_user_id}"
                        )
            else:
                logger.warning(
                    f"Unauthorized access attempt by user {user_authorized} to a document with user_id {doc_user_id}"
                )

        return authorized_documents

    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in query_embeddings_by_file_id | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error in query embeddings | File ID: %s | Query: %s | Error: %s | Traceback: %s",
            body.file_id,
            body.query,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


def generate_digest(page_content: str):
    hash_obj = hashlib.md5(page_content.encode())
    return hash_obj.hexdigest()


# create batches for parallel execution
def create_batches(lst, n):
    """
    Splits the list `lst` into `n` parts.
    The parts will be as evenly sized as possible.
    """
    length = len(lst)
    # Base size of each sublist
    base = length // n
    # How many lists need an extra element
    extra = length % n

    sublists = []
    start_index = 0

    for i in range(n):
        # Determine how many elements this sublist should have
        # Give 1 extra element to the first `extra` sublists
        size = base + (1 if i < extra else 0)
        end_index = start_index + size
        sublists.append(lst[start_index:end_index])
        start_index = end_index

    return sublists


# Define an async function to add documents
async def add_documents_async(pgvector_store, documents, docids):
    if len(documents) > 0:
        logger.info("Adding Documents...")
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"Memory usage-a: {mem_mb} MB")

        nroddocs = await pgvector_store.aadd_documents(documents, ids=docids)
        logger.info(f"Adding Documents done: {len(nroddocs)}")
        mem_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"Memory usage-b: {mem_mb} MB")
        return len(nroddocs)
    else:
        logger.info("No Documents to add = 0...")
        return 0


async def store_data_in_vector_db(
    data: Iterable[Document],
    file_id: str,
    user_id: str = "",
    clean_content: bool = False,
    storage_metadata: Optional[dict] = None,
) -> bool:

    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    logger.info(f"Memory usage1: {mem_mb} MB")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    documents = text_splitter.split_documents(data)

    mem_mb = process.memory_info().rss / 1024 / 1024
    logger.info(f"Memory usage2: {mem_mb} MB")

    # If `clean_content` is True, clean the page_content of each document (remove null bytes)
    if clean_content:
        for doc in documents:
            doc.page_content = clean_text(doc.page_content)

    mem_mb = process.memory_info().rss / 1024 / 1024
    logger.info(f"Memory usage3: {mem_mb} MB")

    # Preparing documents with page content and metadata for insertion.
    storage_meta_to_add = (
        {
            "source": storage_metadata["path"],
            "storage_type": "local",
            "storage_key": storage_metadata["key"],
            "storage_folder": storage_metadata["folder"],
            "original_filename": storage_metadata.get("original_filename"),
        }
        if storage_metadata and storage_metadata.get("storage_type") == "local"
        else (
            {
                "source": f"s3://{storage_metadata['bucket']}/{storage_metadata['key']}",
                "storage_type": "s3",
                "s3_bucket": storage_metadata["bucket"],
                "s3_key": storage_metadata["key"],
                "original_filename": storage_metadata.get("original_filename"),
            }
            if storage_metadata
            else {}
        )
    )
    docs = [
        Document(
            page_content=doc.page_content,
            metadata={
                "file_id": file_id,
                "user_id": user_id,
                "digest": generate_digest(doc.page_content),
                **(doc.metadata or {}),
                # Add storage metadata if available
                **storage_meta_to_add,
            },
        )
        for doc in documents
    ]

    mem_mb = process.memory_info().rss / 1024 / 1024
    logger.info(f"Memory usage4: {mem_mb} MB")

    batches = []
    # Use single batch for simplicity
    batches.append(docs)

    mem_mb = process.memory_info().rss / 1024 / 1024
    logger.info(f"Memory usage5: {mem_mb} MB")

    logger.info(
        f"user: {user_id} with file {file_id} - chunked documents: {len(documents)}"
    )

    tasks = []
    id = 0
    for batch in batches:
        tasks.append(
            add_documents_async(vector_store, batch, [file_id] * len(batch))
        )
        logger.info(
            f"user: {user_id} with file {file_id} - batch {id}: {len(batch)}"
        )
        id += 1

    try:
        ids = 0
        if isinstance(vector_store, AsyncPgVector):

            logger.info(
                f"user: {user_id} with file {file_id} - Execute Getting Documents in Parallel"
            )

            results = await asyncio.gather(*tasks)

            id = 0
            for result in results:
                logger.info(
                    f"user: {user_id} with file {file_id} - Execute Parallel done batch {id} - {result} documents added"
                )
                ids = ids + result
                id += 1

            logger.info(
                f"user: {user_id} with file {file_id} - Execute Parallel done total {ids} documents added"
            )

        else:
            ids = vector_store.add_documents(
                docs, ids=[file_id] * len(documents)
            )

        return {"message": "Documents added successfully", "ids": ids}

    except Exception as e:
        logger.error(
            "Failed to store data in vector DB | File ID: %s | User ID: %s | Error: %s | Traceback: %s",
            file_id,
            user_id,
            str(e),
            traceback.format_exc(),
        )
        return {
            "message": "An error occurred while adding documents.",
            "error": str(e),
        }


@router.post("/local/embed")
async def embed_local_file(
    document: StoreDocument, request: Request, entity_id: str = None
):
    # Check if the file exists
    if not os.path.exists(document.filepath):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.FILE_NOT_FOUND,
        )

    if not hasattr(request.state, "user"):
        user_id = entity_id if entity_id else "public"
    else:
        user_id = entity_id if entity_id else request.state.user.get("id")

    try:
        loader, known_type, file_ext = get_loader(
            document.filename, document.file_content_type, document.filepath
        )
        data = await run_in_executor(None, loader.load)
        result = await store_data_in_vector_db(
            data, document.file_id, user_id, clean_content=file_ext == "pdf"
        )

        if result:
            return {
                "status": True,
                "file_id": document.file_id,
                "filename": document.filename,
                "known_type": known_type,
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=ERROR_MESSAGES.DEFAULT(),
            )
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in embed_local_file | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(e)
        if "No pandoc was found" in str(e):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.PANDOC_NOT_INSTALLED,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT(e),
            )


@router.post("/embed")
async def embed_file(
    request: Request,
    file_id: str = Form(...),
    file: UploadFile = File(...),
    entity_id: str = Form(None),
    agentID: Optional[str] = Form(None),
):
    response_status = True
    response_message = "File processed successfully."
    known_type = None
    storage_metadata = None

    if not hasattr(request.state, "user"):
        user_id = entity_id if entity_id else "public"
    else:
        user_id = entity_id if entity_id else request.state.user.get("id")

    temp_base_path = os.path.join(RAG_UPLOAD_DIR, user_id)
    os.makedirs(temp_base_path, exist_ok=True)
    temp_file_path = os.path.join(RAG_UPLOAD_DIR, user_id, file.filename)

    try:
        async with aiofiles.open(temp_file_path, "wb") as temp_file:
            chunk_size = 64 * 1024  # 64 KB
            while content := await file.read(chunk_size):
                await temp_file.write(content)
    except Exception as e:
        logger.error(
            "Failed to save uploaded file | Path: %s | Error: %s | Traceback: %s",
            temp_file_path,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save the uploaded file. Error: {str(e)}",
        )

    try:
        loader, known_type, file_ext = get_loader(
            file.filename, file.content_type, temp_file_path
        )

        logger.debug(
            f"Loading Filename:{file.filename} - ContentType:{file.content_type} - "
            f"FileExt:{file_ext} - KnownType:{known_type} - Loader:{loader}"
        )

        data = loader.load()

        # Store file in persistent storage if service is available
        if file_storage_service:
            try:
                folder_name = file_storage_service.get_folder_name(
                    user_id, agentID
                )
                storage_key = file_storage_service.generate_storage_key(
                    folder_name, file.filename, file_id
                )
                storage_metadata = await file_storage_service.store_file(
                    temp_file_path,
                    storage_key,
                    file.content_type,
                    file.filename,
                )
                storage_type = "S3" if file_storage_service.use_s3 else "local"
                logger.info(f"File stored in {storage_type}: {storage_key}")
            except Exception as e:
                logger.error(f"File storage failed: {e}")

        result = await store_data_in_vector_db(
            data=data,
            file_id=file_id,
            user_id=user_id,
            clean_content=file_ext == "pdf",
            storage_metadata=storage_metadata,
        )

        if not result:
            # Clean up stored file if database operation failed
            if storage_metadata and file_storage_service:
                try:
                    await file_storage_service.delete_file(
                        storage_metadata.get("key")
                    )
                    logger.info(
                        f"Cleaned up stored file: {storage_metadata.get('key')}"
                    )
                except Exception:
                    logger.error("Failed to cleanup stored file")

            response_status = False
            response_message = "Failed to process/store the file data."
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process/store the file data.",
            )
        elif "error" in result:
            # Clean up stored file if database operation failed
            if storage_metadata and file_storage_service:
                try:
                    await file_storage_service.delete_file(
                        storage_metadata.get("key")
                    )
                    logger.info(
                        f"Cleaned up stored file: {storage_metadata.get('key')}"
                    )
                except Exception:
                    logger.error("Failed to cleanup stored file")

            response_status = False
            response_message = "Failed to process/store the file data."
            if isinstance(result["error"], str):
                response_message = result["error"]
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="An unspecified error occurred.",
                )
    except HTTPException as http_exc:
        response_status = False
        response_message = f"HTTP Exception: {http_exc.detail}"
        logger.error(
            "HTTP Exception in embed_file | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        # Clean up stored file on processing failure
        if storage_metadata and file_storage_service:
            try:
                await file_storage_service.delete_file(
                    storage_metadata.get("key")
                )
                logger.info(
                    f"Cleaned up stored file: {storage_metadata.get('key')}"
                )
            except Exception:
                logger.error("Failed to cleanup stored file")

        response_status = False
        response_message = f"Error during file processing: {str(e)}"
        logger.error(
            "Error during file processing: %s\nTraceback: %s",
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error during file processing: {str(e)}",
        )
    finally:
        try:
            await aiofiles.os.remove(temp_file_path)
        except Exception as e:
            logger.error(
                "Failed to remove temporary file | Path: %s | Error: %s | Traceback: %s",
                temp_file_path,
                str(e),
                traceback.format_exc(),
            )

    return {
        "status": response_status,
        "message": response_message,
        "file_id": file_id,
        "filename": file.filename,
        "known_type": known_type,
    }


@router.get("/documents/{id}/context")
async def load_document_context(id: str):
    ids = [id]
    try:
        if isinstance(vector_store, AsyncPgVector):
            existing_ids = await vector_store.get_filtered_ids(ids)
            documents = await vector_store.get_documents_by_ids(ids)
        else:
            existing_ids = vector_store.get_filtered_ids(ids)
            documents = vector_store.get_documents_by_ids(ids)

        # Ensure the requested id exists
        if not all(id in existing_ids for id in ids):
            raise HTTPException(
                status_code=404, detail="The specified file_id was not found"
            )

        # Ensure documents list is not empty
        if not documents:
            raise HTTPException(
                status_code=404, detail="No document found for the given ID"
            )

        return process_documents(documents)
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in load_document_context | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error loading document context | Document ID: %s | Error: %s | Traceback: %s",
            id,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


@router.post("/embed-upload")
async def embed_file_upload(
    request: Request,
    file_id: str = Form(...),
    uploaded_file: UploadFile = File(...),
    entity_id: str = Form(None),
):
    temp_file_path = os.path.join(RAG_UPLOAD_DIR, uploaded_file.filename)

    if not hasattr(request.state, "user"):
        user_id = entity_id if entity_id else "public"
    else:
        user_id = entity_id if entity_id else request.state.user.get("id")

    try:
        with open(temp_file_path, "wb") as temp_file:
            copyfileobj(uploaded_file.file, temp_file)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save the uploaded file. Error: {str(e)}",
        )

    try:
        loader, known_type, file_ext = get_loader(
            uploaded_file.filename, uploaded_file.content_type, temp_file_path
        )

        data = await run_in_executor(None, loader.load)
        result = await store_data_in_vector_db(
            data, file_id, user_id, clean_content=file_ext == "pdf"
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process/store the file data.",
            )
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in embed_file_upload | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error during file processing | File: %s | Error: %s | Traceback: %s",
            uploaded_file.filename,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error during file processing: {str(e)}",
        )
    finally:
        os.remove(temp_file_path)

    return {
        "status": True,
        "message": "File processed successfully.",
        "file_id": file_id,
        "filename": uploaded_file.filename,
        "known_type": known_type,
    }


@router.post("/query_multiple")
async def query_embeddings_by_file_ids(body: QueryMultipleBody):
    try:
        # Get the embedding of the query text
        embedding = get_cached_query_embedding(body.query)

        # Perform similarity search with the query embedding and filter by the file_ids in metadata
        if isinstance(vector_store, AsyncPgVector):
            documents = await run_in_executor(
                None,
                vector_store.similarity_search_with_score_by_vector,
                embedding,
                k=body.k,
                filter={"file_id": {"$in": body.file_ids}},
            )
        else:
            documents = vector_store.similarity_search_with_score_by_vector(
                embedding, k=body.k, filter={"file_id": {"$in": body.file_ids}}
            )

        # Ensure documents list is not empty
        if not documents:
            raise HTTPException(
                status_code=404, detail="No documents found for the given query"
            )

        return documents
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in query_embeddings_by_file_ids | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error in query multiple embeddings | File IDs: %s | Query: %s | Error: %s | Traceback: %s",
            body.file_ids,
            body.query,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))
