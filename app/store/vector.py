# vector.py
from typing import List, Optional, Tuple, Any
import copy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from sqlalchemy.orm import Session
from sqlalchemy import delete
from langchain_community.vectorstores.pgvector import PGVector
from langchain_mongodb import MongoDBAtlasVectorSearch

class ExtendedPgVector(PGVector):
    def get_all_ids(self) -> list[str]:
        with Session(self._bind) as session:
            results = session.query(self.EmbeddingStore.custom_id).all()
            return [result[0] for result in results if result[0] is not None]

    def get_documents_by_ids(self, ids: list[str]) -> list[Document]:
        with Session(self._bind) as session:
            results = session.query(self.EmbeddingStore).filter(
                self.EmbeddingStore.custom_id.in_(ids)
            ).all()
            return [
                Document(page_content=result.document, metadata=result.cmetadata or {})
                for result in results if result.custom_id in ids
            ]

    def _delete_multiple(self, ids: Optional[list[str]] = None, collection_only: bool = False) -> None:
        with Session(self._bind) as session:
            if ids is not None:
                self.logger.debug("Trying to delete vectors by ids (using custom ids)")
                stmt = delete(self.EmbeddingStore)
                if collection_only:
                    collection = self.get_collection(session)
                    if not collection:
                        self.logger.warning("Collection not found")
                        return
                    stmt = stmt.where(self.EmbeddingStore.collection_id == collection.uuid)
                stmt = stmt.where(self.EmbeddingStore.custom_id.in_(ids))
                session.execute(stmt)
            session.commit()

class AsyncPgVector(ExtendedPgVector):
    async def get_all_ids(self) -> list[str]:
        return await run_in_executor(None, super().get_all_ids)

    async def get_documents_by_ids(self, ids: list[str]) -> list[Document]:
        return await run_in_executor(None, super().get_documents_by_ids, ids)

    async def delete(self, ids: Optional[list[str]] = None, collection_only: bool = False) -> None:
        await run_in_executor(None, self._delete_multiple, ids, collection_only)

class AtlasMongoVector(MongoDBAtlasVectorSearch):
    @property
    def embedding_function(self) -> Embeddings:
        return self.embeddings

    def add_documents(self, docs: list[Document], ids: list[str]):
        # Use 'i' instead of 'id' to avoid shadowing the built-in function
        new_ids = [i for i in range(len(ids))]
        file_id = docs[0].metadata['file_id']
        f_ids = [f'{file_id}_{i}' for i in new_ids]
        return super().add_documents(docs, f_ids)

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        docs = self._similarity_search_with_score(
            embedding, k=k, pre_filter=filter, post_filter_pipeline=None, **kwargs
        )
        processed_documents: List[Tuple[Document, float]] = []
        for document, score in docs:
            doc_copy = copy.deepcopy(document.__dict__)
            # Remove MongoDB's _id from metadata if present
            if "metadata" in doc_copy and "_id" in doc_copy["metadata"]:
                del doc_copy["metadata"]["_id"]
            new_document = Document(**doc_copy)
            processed_documents.append((new_document, score))
        return processed_documents

    def get_all_ids(self) -> list[str]:
        return self._collection.distinct("file_id")

    def get_documents_by_ids(self, ids: list[str]) -> list[Document]:
        return [
            Document(
                page_content=doc["text"],
                metadata={
                    "file_id": doc["file_id"],
                    "user_id": doc["user_id"],
                    "digest": doc["digest"],
                    "source": doc["source"],
                    "page": int(doc.get("page", 0)),
                },
            )
            for doc in self._collection.find({"file_id": {"$in": ids}})
        ]

    def delete(self, ids: Optional[list[str]] = None) -> None:
        if ids is not None:
            self._collection.delete_many({"file_id": {"$in": ids}})

def get_vector_store(
    connection_string: str,
    embeddings: Embeddings,
    collection_name: str,
    mode: str = "sync",
    search_index: Optional[str] = None
):
    if mode == "sync":
        return ExtendedPgVector(
            connection_string=connection_string,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
    elif mode == "async":
        return AsyncPgVector(
            connection_string=connection_string,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
    elif mode == "atlas-mongo":
        from pymongo import MongoClient
        mongo_db = MongoClient(connection_string).get_database()
        mong_collection = mongo_db[collection_name]
        return AtlasMongoVector(
            collection=mong_collection,
            embedding=embeddings,
            index_name=search_index
        )
    else:
        raise ValueError("Invalid mode specified. Choose 'sync', 'async', or 'atlas-mongo'.")