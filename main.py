import os

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException
from langchain.schema import Document
from langchain_core.runnables.config import run_in_executor

from models import DocumentModel, DocumentResponse
from store import AsyncPgVector

from config import pgvector_store

load_dotenv(find_dotenv())

app = FastAPI()


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
            else pgvector_store.add_documents(docs)
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

        return ids
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
