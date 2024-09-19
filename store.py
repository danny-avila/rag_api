from typing import Any, Optional
from aiohttp import Payload
from sqlalchemy import delete
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.documents import Document
from langchain_core.runnables.config import run_in_executor
from sqlalchemy.orm import Session
from langchain_community.vectorstores import Qdrant
import qdrant_client as client
from qdrant_client.http import models



from langchain_mongodb import MongoDBAtlasVectorSearch
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
            self,
            ids: Optional[list[str]] = None,
            collection_only: bool = False
        ) -> None:
            await run_in_executor(None, self._delete_multiple, ids, collection_only)

class ExtendedQdrant(Qdrant):
    def delete_vectors_by_source_document(self, source_document_ids: list[str]) -> None:
        points_selector = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.file_id",
                    match=models.MatchAny(any=source_document_ids),
                ),
            ],
        )
        response = self.client.delete(collection_name=self.collection_name, points_selector=points_selector)
        status = response.status.name
        return status
      
    
    def get_all_ids(self) -> list[str]:
        collection_info = self.client.get_collection(self.collection_name)
        total_points = collection_info.points_count

        # Scroll through all points in the collection
        unique_values = set()
        pointsRec = 0
        limit = 500  #How much to load each time
        next_offset = None
        while pointsRec < total_points:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=True,
                offset=next_offset,
            )
            for point in points:
                unique_values.add(point.payload['metadata']['file_id'])
            pointsRec += limit
        return list(unique_values)     

    def get_documents_by_ids(self, ids: list[str]):
        filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.file_id",
                    match=models.MatchAny(any=ids)
                )
            ]
        )
        limit = 500
        next_offset = None
        docList = []
        while True:
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter,
                limit=limit,
                offset=next_offset
            )
            points, next_offset = results
            if points:
                docList.extend([Document(page_content=point.payload['page_content'],
                         metadata=point.payload['metadata']
                        ) 
                           for point in points 
                           ])
            if not next_offset:
                break
        return docList
            
        

        
class AsyncQdrant(ExtendedQdrant):
    async def get_all_ids(self) -> list[str]:
        return await run_in_executor(None, super().get_all_ids)

    async def get_documents_by_ids(self, ids: list[str]) -> list[Document]:
        return await run_in_executor(None, super().get_documents_by_ids, ids)

    async def delete(
        self,
        ids: Optional[list[str]] = None
    ) -> None:
        # Garantir que o argumento correto está sendo passado
        await run_in_executor(None, self.delete_vectors_by_source_document, ids)

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


async_DB = (AsyncPgVector, AsyncQdrant) #Add if async database implementation 
