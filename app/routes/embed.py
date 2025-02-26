import os
import traceback
import hashlib
from shutil import copyfileobj
import aiofiles
import aiofiles.os
from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException, status

# Import the proper configuration values.
from app.config import logger, CHUNK_SIZE, CHUNK_OVERLAP, RAG_UPLOAD_DIR, vector_store, known_source_ext
from app.constants import ERROR_MESSAGES
from app.models import StoreDocument
from app.parsers import clean_text

from app.store.vector import AsyncPgVector  # Ensure that AsyncPgVector now includes aadd_documents

router = APIRouter()


def generate_digest(page_content: str):
    return hashlib.md5(page_content.encode()).hexdigest()


async def store_data_in_vector_db(data, file_id: str, user_id: str = "", clean_content: bool = False):
    # Import text splitter and Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document

    # Use CHUNK_SIZE and CHUNK_OVERLAP (which should be integers) instead of RAG_UPLOAD_DIR
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
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
        # If using AsyncPgVector, call the asynchronous add_documents method.
        if isinstance(vector_store, AsyncPgVector):
            # Note: Ensure that AsyncPgVector has an asynchronous wrapper (aadd_documents)
            ids = await vector_store.aadd_documents(docs, ids=[file_id] * len(documents))
        else:
            ids = vector_store.add_documents(docs, ids=[file_id] * len(documents))
        return {"message": "Documents added successfully", "ids": ids}
    except Exception as e:
        logger.error(
            "Failed to store data in vector DB | File ID: %s | User ID: %s | Error: %s | Traceback: %s",
            file_id, user_id, str(e), traceback.format_exc()
        )
        return {"message": "An error occurred while adding documents.", "error": str(e)}


def get_loader(filename: str, file_content_type: str, filepath: str):
    file_ext = filename.split(".")[-1].lower()
    known_type = True
    if file_ext == "pdf":
        # Using new import location is recommended to avoid deprecation warnings.
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(filepath, extract_images=os.getenv("PDF_EXTRACT_IMAGES", "False").lower() == "true")
    elif file_ext == "csv":
        from langchain.document_loaders import CSVLoader
        loader = CSVLoader(filepath)
    elif file_ext == "rst":
        from langchain.document_loaders import UnstructuredRSTLoader
        loader = UnstructuredRSTLoader(filepath, mode="elements")
    elif file_ext == "xml":
        from langchain.document_loaders import UnstructuredXMLLoader
        loader = UnstructuredXMLLoader(filepath)
    elif file_ext == "pptx":
        from langchain.document_loaders import UnstructuredPowerPointLoader
        loader = UnstructuredPowerPointLoader(filepath)
    elif file_ext == "md":
        from langchain.document_loaders import UnstructuredMarkdownLoader
        loader = UnstructuredMarkdownLoader(filepath)
    elif file_content_type == "application/epub+zip":
        from langchain.document_loaders import UnstructuredEPubLoader
        loader = UnstructuredEPubLoader(filepath)
    elif file_content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or file_ext in [
        "doc", "docx"]:
        from langchain.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(filepath)
    elif file_content_type in ["application/vnd.ms-excel",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"] or file_ext in [
        "xls", "xlsx"]:
        from langchain.document_loaders import UnstructuredExcelLoader
        loader = UnstructuredExcelLoader(filepath)
    elif file_content_type == "application/json" or file_ext == "json":
        from langchain.document_loaders import TextLoader
        loader = TextLoader(filepath, autodetect_encoding=True)
    elif file_ext in known_source_ext or (file_content_type and "text/" in file_content_type):
        from langchain.document_loaders import TextLoader
        loader = TextLoader(filepath, autodetect_encoding=True)
    else:
        from langchain.document_loaders import TextLoader
        loader = TextLoader(filepath, autodetect_encoding=True)
        known_type = False
    return loader, known_type, file_ext


@router.post("/local/embed")
async def embed_local_file(document: StoreDocument, request: Request, entity_id: str = None):
    if not os.path.exists(document.filepath):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.FILE_NOT_FOUND)
    if not hasattr(request.state, "user"):
        user_id = entity_id if entity_id else "public"
    else:
        user_id = entity_id if entity_id else request.state.user.get("id")
    try:
        loader, known_type, _ = get_loader(document.filename, document.file_content_type, document.filepath)
        data = loader.load()
        result = await store_data_in_vector_db(data, document.file_id, user_id)
        if result:
            return {"status": True, "file_id": document.file_id, "filename": document.filename,
                    "known_type": known_type}
        else:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=ERROR_MESSAGES.DEFAULT())
    except Exception as e:
        logger.error(e)
        if "No pandoc was found" in str(e):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.PANDOC_NOT_INSTALLED)
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/embed")
async def embed_file(request: Request, file_id: str = Form(...), file: UploadFile = File(...),
                     entity_id: str = Form(None)):
    response_status = True
    response_message = "File processed successfully."
    known_type = None
    if not hasattr(request.state, "user"):
        user_id = entity_id if entity_id else "public"
    else:
        user_id = entity_id if entity_id else request.state.user.get("id")
    temp_base_path = os.path.join(RAG_UPLOAD_DIR, user_id)
    os.makedirs(temp_base_path, exist_ok=True)
    temp_file_path = os.path.join(temp_base_path, file.filename)
    try:
        async with aiofiles.open(temp_file_path, "wb") as temp_file:
            chunk_size = 64 * 1024
            while content := await file.read(chunk_size):
                await temp_file.write(content)
    except Exception as e:
        logger.error("Failed to save uploaded file | Path: %s | Error: %s | Traceback: %s", temp_file_path, str(e),
                     traceback.format_exc())
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to save the uploaded file. Error: {str(e)}")
    try:
        loader, known_type, file_ext = get_loader(file.filename, file.content_type, temp_file_path)
        data = loader.load()
        result = await store_data_in_vector_db(data=data, file_id=file_id, user_id=user_id,
                                               clean_content=(file_ext == "pdf"))
        if not result:
            response_status = False
            response_message = "Failed to process/store the file data."
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Failed to process/store the file data.")
        elif "error" in result:
            response_status = False
            response_message = result["error"] if isinstance(result["error"], str) else "An unspecified error occurred."
    except Exception as e:
        response_status = False
        response_message = f"Error during file processing: {str(e)}"
        logger.error("Error during file processing: %s\nTraceback: %s", str(e), traceback.format_exc())
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error during file processing: {str(e)}")
    finally:
        try:
            await aiofiles.os.remove(temp_file_path)
        except Exception as e:
            logger.error("Failed to remove temporary file | Path: %s | Error: %s | Traceback: %s", temp_file_path,
                         str(e), traceback.format_exc())
    return {"status": response_status, "message": response_message, "file_id": file_id, "filename": file.filename,
            "known_type": known_type}


@router.post("/embed-upload")
async def embed_file_upload(request: Request, file_id: str = Form(...), uploaded_file: UploadFile = File(...),
                            entity_id: str = Form(None)):
    temp_file_path = os.path.join(RAG_UPLOAD_DIR, uploaded_file.filename)
    if not hasattr(request.state, "user"):
        user_id = entity_id if entity_id else "public"
    else:
        user_id = entity_id if entity_id else request.state.user.get("id")
    try:
        with open(temp_file_path, "wb") as temp_file:
            copyfileobj(uploaded_file.file, temp_file)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to save the uploaded file. Error: {str(e)}")
    try:
        loader, known_type, _ = get_loader(uploaded_file.filename, uploaded_file.content_type, temp_file_path)
        data = loader.load()
        result = await store_data_in_vector_db(data, file_id, user_id)
        if not result:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Failed to process/store the file data.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error during file processing: {str(e)}")
    finally:
        os.remove(temp_file_path)
    return {"status": True, "message": "File processed successfully.", "file_id": file_id,
            "filename": uploaded_file.filename, "known_type": known_type}