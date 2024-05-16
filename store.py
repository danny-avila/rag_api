from typing import Any, Optional
from sqlalchemy import delete
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.documents import Document
from langchain_core.runnables.config import run_in_executor
from sqlalchemy.orm import Session
import pinecone
from langchain_pinecone._utilities import DistanceStrategy
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_pinecone import PineconeVectorStore, Pinecone
from langchain_core.embeddings import Embeddings
from typing import (
    List,
    Optional,
    Tuple,
)
import copy


class ExtendedPgVector(PGVector):

    def get_all_ids(self) -> list[str]:
        with Session(self._bind) as session:
            results = session.query(self.EmbeddingStore.custom_id).all()
            return [result[0] for result in results if result[0] is not None]

    def get_documents_by_ids(self, ids: list[str]) -> list[Document]:
        with Session(self._bind) as session:
            results = (
                session.query(self.EmbeddingStore)
                .filter(self.EmbeddingStore.custom_id.in_(ids))
                .all()
            )
            return [
                Document(page_content=result.document, metadata=result.cmetadata or {})
                for result in results
                if result.custom_id in ids
            ]

    def _delete_multiple(
        self, ids: Optional[list[str]] = None, collection_only: bool = False
    ) -> None:
        with Session(self._bind) as session:
            if ids is not None:
                self.logger.debug(
                    "Trying to delete vectors by ids (represented by the model "
                    "using the custom ids field)"
                )

                stmt = delete(self.EmbeddingStore)

                if collection_only:
                    collection = self.get_collection(session)
                    if not collection:
                        self.logger.warning("Collection not found")
                        return

                    stmt = stmt.where(
                        self.EmbeddingStore.collection_id == collection.uuid
                    )

                stmt = stmt.where(self.EmbeddingStore.custom_id.in_(ids))
                session.execute(stmt)
            session.commit()


class AsyncPgVector(ExtendedPgVector):

    async def get_all_ids(self) -> list[str]:
        return await run_in_executor(None, super().get_all_ids)

    async def get_documents_by_ids(self, ids: list[str]) -> list[Document]:
        return await run_in_executor(None, super().get_documents_by_ids, ids)

    async def delete(
        self, ids: Optional[list[str]] = None, collection_only: bool = False
    ) -> None:
        await run_in_executor(None, self._delete_multiple, ids, collection_only)


class AtlasMongoVector(MongoDBAtlasVectorSearch):
    @property
    def embedding_function(self) -> Embeddings:
        return self.embeddings

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        docs = self._similarity_search_with_score(
            embedding,
            k=k,
            pre_filter=filter,
            post_filter_pipeline=None,
            **kwargs,
        )
        # remove `metadata._id` since MongoDB ObjectID is not serializable
        # Process the documents to remove metadata._id
        processed_documents: List[Tuple[Document, float]] = []
        for document, score in docs:
            # Make a deep copy of the document to avoid mutating the original
            doc_copy = copy.deepcopy(
                document.__dict__
            )  # If Document is a dataclass or similar; adjust as needed

            # Remove _id field from metadata if it exists
            if "metadata" in doc_copy and "_id" in doc_copy["metadata"]:
                del doc_copy["metadata"]["_id"]

            # Create a new Document instance without the _id
            new_document = Document(
                **doc_copy
            )  # Adjust this line according to how you instantiate your Document

            # Append the new document and score to the list as a tuple
            processed_documents.append((new_document, score))
        return processed_documents

    def get_all_ids(self) -> list[str]:
        # implement the return of unique file_id fields in self._collection
        return self._collection.distinct("file_id")

    def get_documents_by_ids(self, ids: list[str]) -> list[Document]:
        # implement the return of documents by file_id in self._collection

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
        # implement the deletion of documents by file_id in self._collection
        if ids is not None:
            self._collection.delete_many({"file_id": {"$in": ids}})

class PineconeVector(PineconeVectorStore):
    @property
    def embedding_function(self) -> Embeddings:
        return self.embeddings
    
    def __init__(self, embedding: Embeddings, api_key: str, index_name: str, namespace: Optional[str] = None):
        self.index_name = index_name
        self.namespace = namespace
        super().__init__(index_name=self.index_name, embedding=embedding, text_key="text", namespace=namespace, distance_strategy=DistanceStrategy.COSINE, pinecone_api_key=api_key)

    def get_all_ids(self) -> List[str]:
        """
        Retrieve all vector IDs in the Pinecone index.
        """
        return self._index.list(self.namespace)

    def get_documents_by_ids(self, ids: List[str]) -> List[Document]:
        """
        Retrieve documents by their IDs from the Pinecone index.
        """
        results = self._index.fetch(ids, namespace=self.namespace)
        documents = []
        for result in results['vectors'].values():
            metadata = result['metadata']
            doc = Document(page_content=metadata['text'], metadata=metadata)
            documents.append(doc)
        return documents

    def delete(self, ids: Optional[List[str]] = None) -> None:
        """
        Delete vectors by their IDs from the Pinecone index.
        """
        if ids:
            self._index.delete(ids, namespace=self.namespace)

    def similarity_search_with_score_by_vector(
        self, 
        embedding: List[float], 
        k: int = 4, 
        filter: Optional[dict] = None, 
        **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """
        Perform a similarity search with scores using an embedding vector.
        """
        query_results = self._index.query(embedding, top_k=k, include_metadata=True, namespace=self.namespace, **kwargs)
        docs = query_results['matches']
        processed_documents = []
        for match in docs:
            metadata = match['metadata']
            if 'metadata' in metadata and '_id' in metadata['metadata']:
                del metadata['metadata']['_id']
            doc = Document(page_content=metadata['text'], metadata=metadata)
            processed_documents.append((doc, match['score']))
        return processed_documents