import os

import hashlib
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from langchain.schema import Document
from langchain_core.runnables.config import run_in_executor

from models import DocumentModel, DocumentResponse, StoreDocument
from store import AsyncPgVector

load_dotenv(find_dotenv())

from config import (
    PDF_EXTRACT_IMAGES,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    pgvector_store,
    known_source_ext,
    # RAG_EMBEDDING_MODEL,
    # RAG_EMBEDDING_MODEL_DEVICE_TYPE,
    # RAG_TEMPLATE,
)

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
from langchain.text_splitter import RecursiveCharacterTextSplitter

from constants import ERROR_MESSAGES

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.CHUNK_SIZE = CHUNK_SIZE
app.state.CHUNK_OVERLAP = CHUNK_OVERLAP
app.state.PDF_EXTRACT_IMAGES = PDF_EXTRACT_IMAGES

def get_env_variable(var_name: str) -> str:
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"Environment variable '{var_name}' not found.")
    return value

@app.post("/add-documents/")
async def add_documents(documents: list[DocumentModel]):
    try:
        docs = [
            Document(
                page_content=doc.page_content,
                metadata={
                    "file_id": doc.id,
                    "digest": doc.generate_digest(),
                    **(doc.metadata or {}),
                },
            )
            for doc in documents
        ]
        ids = (
            await pgvector_store.aadd_documents(docs, ids=[doc.id for doc in documents])
            if isinstance(pgvector_store, AsyncPgVector)
            else pgvector_store.add_documents(docs, ids=[doc.id for doc in documents])
        )
        return {"message": "Documents added successfully", "ids": ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-all-ids/")
async def get_all_ids():
    try:
        if isinstance(pgvector_store, AsyncPgVector):
            ids = await pgvector_store.get_all_ids()
        else:
            ids = pgvector_store.get_all_ids()

        return list(set(ids))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get-documents-by-ids/", response_model=list[DocumentResponse])
async def get_documents_by_ids(ids: list[str]):
    try:
        if isinstance(pgvector_store, AsyncPgVector):
            existing_ids = await pgvector_store.get_all_ids()
            documents = await pgvector_store.get_documents_by_ids(ids)
        else:
            existing_ids = pgvector_store.get_all_ids()
            documents = pgvector_store.get_documents_by_ids(ids)

        if not all(id in existing_ids for id in ids):
            raise HTTPException(status_code=404, detail="One or more IDs not found")

        return documents
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete-documents/")
async def delete_documents(ids: list[str]):
    try:
        if isinstance(pgvector_store, AsyncPgVector):
            existing_ids = await pgvector_store.get_all_ids()
            await pgvector_store.delete(ids=ids)
        else:
            existing_ids = pgvector_store.get_all_ids()
            pgvector_store.delete(ids=ids)

        if not all(id in existing_ids for id in ids):
            raise HTTPException(status_code=404, detail="One or more IDs not found")

        return {"message": f"{len(ids)} documents deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query-embeddings-by-file-id/")
async def query_embeddings_by_file_id(file_id: str, query: str, k: int = 4):
    try:
        # Get the embedding of the query text
        embedding = pgvector_store.embedding_function.embed_query(query)

        # Perform similarity search with the query embedding and filter by the file_id in metadata
        if isinstance(pgvector_store, AsyncPgVector):
            documents = await run_in_executor(
                None,
                pgvector_store.similarity_search_with_score_by_vector,
                embedding,
                k=k,
                filter={"file_id": file_id}
            )
        else:
            documents = pgvector_store.similarity_search_with_score_by_vector(
                embedding,
                k=k,
                filter={"file_id": file_id}
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
        if isinstance(pgvector_store, AsyncPgVector):
            ids = await pgvector_store.aadd_documents(docs, ids=[file_id]*len(documents))
        else:
            ids = pgvector_store.add_documents(docs, ids=[file_id]*len(documents))
        
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

@app.post("/doc")
async def store_doc(document: StoreDocument):
    
    # Check if the file exists
    if not os.path.exists(document.filepath):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.FILE_NOT_FOUND(),
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
