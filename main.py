import os
import hashlib
from shutil import copyfileobj
from langchain.schema import Document
from contextlib import asynccontextmanager
from dotenv import find_dotenv, load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, status
from langchain_core.runnables.config import run_in_executor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    WebBaseLoader,
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredEPubLoader,
    UnstructuredMarkdownLoader,
    UnstructuredXMLLoader,
    UnstructuredRSTLoader,
    UnstructuredExcelLoader,
)

from models import DocumentResponse, StoreDocument, QueryRequestBody, QueryMultipleBody
from psql import PSQLDatabase, ensure_custom_id_index_on_embedding
from middleware import security_middleware
from pgvector_routes import router as pgvector_router
from parsers import process_documents
from constants import ERROR_MESSAGES
from store import AsyncPgVector

load_dotenv(find_dotenv())

from config import (
    debug_mode,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    vector_store,
    RAG_UPLOAD_DIR,
    known_source_ext,
    PDF_EXTRACT_IMAGES,
    # RAG_EMBEDDING_MODEL,
    # RAG_EMBEDDING_MODEL_DEVICE_TYPE,
    # RAG_TEMPLATE,
)

app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic goes here
    await PSQLDatabase.get_pool()  # Initialize the pool
    await ensure_custom_id_index_on_embedding()
    
    yield  # The application is now up and serving requests

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.middleware("http")(security_middleware)

app.state.CHUNK_SIZE = CHUNK_SIZE
app.state.CHUNK_OVERLAP = CHUNK_OVERLAP
app.state.PDF_EXTRACT_IMAGES = PDF_EXTRACT_IMAGES

@app.get("/ids")
async def get_all_ids():
    try:
        if isinstance(vector_store, AsyncPgVector):
            ids = await vector_store.get_all_ids()
        else:
            ids = vector_store.get_all_ids()

        return list(set(ids))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", response_model=list[DocumentResponse])
async def get_documents_by_ids(ids: list[str]):
    try:
        if isinstance(vector_store, AsyncPgVector):
            existing_ids = await vector_store.get_all_ids()
            documents = await vector_store.get_documents_by_ids(ids)
        else:
            existing_ids = vector_store.get_all_ids()
            documents = vector_store.get_documents_by_ids(ids)

        if not all(id in existing_ids for id in ids):
            raise HTTPException(status_code=404, detail="One or more IDs not found")

        return documents
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents")
async def delete_documents(ids: list[str]):
    try:
        if isinstance(vector_store, AsyncPgVector):
            existing_ids = await vector_store.get_all_ids()
            await vector_store.delete(ids=ids)
        else:
            existing_ids = vector_store.get_all_ids()
            vector_store.delete(ids=ids)

        if not all(id in existing_ids for id in ids):
            raise HTTPException(status_code=404, detail="One or more IDs not found")

        file_count = len(ids)
        return {"message": f"Documents for {file_count} file{'s' if file_count > 1 else ''} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_embeddings_by_file_id(body: QueryRequestBody):
    try:
        embedding = vector_store.embedding_function.embed_query(body.query)

        if isinstance(vector_store, AsyncPgVector):
            documents = await run_in_executor(
                None,
                vector_store.similarity_search_with_score_by_vector,
                embedding,
                k=body.k,
                filter={"file_id": body.file_id}
            )
        else:
            documents = vector_store.similarity_search_with_score_by_vector(
                embedding,
                k=body.k,
                filter={"file_id": body.file_id}
            )

        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def generate_digest(page_content: str):
    hash_obj = hashlib.md5(page_content.encode())
    return hash_obj.hexdigest()

async def store_data_in_vector_db(data, file_id, overwrite: bool = False) -> bool:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=app.state.CHUNK_SIZE, chunk_overlap=app.state.CHUNK_OVERLAP
    )
    documents = text_splitter.split_documents(data)


  # Preparing documents with page content and metadata for insertion.
    docs = [
        Document(
            page_content=doc.page_content,
            metadata={
                "file_id": file_id,
                "digest": generate_digest(doc.page_content),
                **(doc.metadata or {}),
            },
        )
        for doc in documents
    ]
    
    try:
        if isinstance(vector_store, AsyncPgVector):
            ids = await vector_store.aadd_documents(docs, ids=[file_id]*len(documents))
        else:
            ids = vector_store.add_documents(docs, ids=[file_id]*len(documents))
        
        return {"message": "Documents added successfully", "ids": ids}

    except Exception as e:
        print(e)
        # Checking if a unique constraint error occurred, to handle overwrite logic if needed.
        if e.__class__.__name__ == "UniqueConstraintError" and overwrite:
            # Functionality to overwrite existing documents.
            # This might require fetching existing document IDs, deleting them, and then re-inserting the documents.
            return {"message": "Documents exist. Overwrite not implemented.", "error": str(e)}
        
        return {"message": "An error occurred while adding documents.", "error": str(e)}

def get_loader(filename: str, file_content_type: str, filepath: str):
    file_ext = filename.split(".")[-1].lower()
    known_type = True

    if file_ext == "pdf":
        loader = PyPDFLoader(filepath, extract_images=app.state.PDF_EXTRACT_IMAGES)
    elif file_ext == "csv":
        loader = CSVLoader(filepath)
    elif file_ext == "rst":
        loader = UnstructuredRSTLoader(filepath, mode="elements")
    elif file_ext == "xml":
        loader = UnstructuredXMLLoader(filepath)
    elif file_ext == "md":
        loader = UnstructuredMarkdownLoader(filepath)
    elif file_content_type == "application/epub+zip":
        loader = UnstructuredEPubLoader(filepath)
    elif (
        file_content_type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        or file_ext in ["doc", "docx"]
    ):
        loader = Docx2txtLoader(filepath)
    elif file_content_type in [
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ] or file_ext in ["xls", "xlsx"]:
        loader = UnstructuredExcelLoader(filepath)
    elif file_ext in known_source_ext or (
        file_content_type and file_content_type.find("text/") >= 0
    ):
        loader = TextLoader(filepath)
    else:
        loader = TextLoader(filepath)
        known_type = False

    return loader, known_type

@app.post("/embed")
async def embed_file(document: StoreDocument):
    
    # Check if the file exists
    if not os.path.exists(document.filepath):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.FILE_NOT_FOUND,
        )

    try:
        loader, known_type = get_loader(document.filename, document.file_content_type, document.filepath)
        data = loader.load()
        result = await store_data_in_vector_db(data, document.file_id)

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
    except Exception as e:
        print(e)
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

@app.get("/documents/{id}/context")
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

        return process_documents(documents)
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )

@app.post("/embed-upload")
async def embed_file_upload(file_id: str = Form(...), uploaded_file: UploadFile = File(...)):
    temp_file_path = os.path.join(RAG_UPLOAD_DIR, uploaded_file.filename)

    try:
        with open(temp_file_path, 'wb') as temp_file:
            copyfileobj(uploaded_file.file, temp_file)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to save the uploaded file. Error: {str(e)}")
    
    try:
        loader, known_type = get_loader(uploaded_file.filename, uploaded_file.content_type, temp_file_path)
        
        data = loader.load()
        result = await store_data_in_vector_db(data, file_id)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process/store the file data.",
            )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Error during file processing: {str(e)}")
    finally:
        os.remove(temp_file_path)
    
    return {
        "status": True,
        "message": "File processed successfully.",
        "file_id": file_id,
        "filename": uploaded_file.filename,
        "known_type": known_type,
    }

@app.post("/query_multiple")
async def query_embeddings_by_file_ids(body: QueryMultipleBody):
    try:
        # Get the embedding of the query text
        embedding = vector_store.embedding_function.embed_query(body.query)

        # Perform similarity search with the query embedding and filter by the file_ids in metadata
        if isinstance(vector_store, AsyncPgVector):
            documents = await run_in_executor(
                None,
                vector_store.similarity_search_with_score_by_vector,
                embedding,
                k=body.k,
                filter={"custom_id": {"$in": body.file_ids}}
            )
        else:
            documents = vector_store.similarity_search_with_score_by_vector(
                embedding,
                k=body.k,
                filter={"custom_id": {"$in": body.file_ids}}
            )

        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if debug_mode:
    app.include_router(router=pgvector_router)
