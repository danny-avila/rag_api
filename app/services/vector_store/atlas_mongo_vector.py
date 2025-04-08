import copy
from typing import Any, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_mongodb import MongoDBAtlasVectorSearch

class AtlasMongoVector(MongoDBAtlasVectorSearch):
    @property
    def embedding_function(self) -> Embeddings:
        return self.embeddings

    def add_documents(self, docs: list[Document], ids: list[str]):
        # {file_id}_{idx}
        new_ids = [id for id in range(len(ids))]
        file_id = docs[0].metadata['file_id']
        f_ids = [f'{file_id}_{id}' for id in new_ids]
        return super().add_documents(docs, f_ids)

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
        processed_documents: List[Tuple[Document, float]] = []
        for document, score in docs:
            # Make a deep copy to avoid mutating the original document
            doc_copy = copy.deepcopy(document.__dict__)
            # Remove _id field from metadata if it exists
            if "metadata" in doc_copy and "_id" in doc_copy["metadata"]:
                del doc_copy["metadata"]["_id"]
            new_document = Document(**doc_copy)
            processed_documents.append((new_document, score))
        return processed_documents

    def get_all_ids(self) -> list[str]:
        # Return unique file_id fields in self._collection
        return self._collection.distinct("file_id")
    
    def get_filtered_ids(self, ids: list[str]) -> list[str]:
        # Return unique file_id fields filtered by the provided ids
        return self._collection.distinct("file_id", {"file_id": {"$in": ids}})

    def get_documents_by_ids(self, ids: list[str]) -> list[Document]:
        # Return documents filtered by file_id
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
        # Delete documents by file_id
        if ids is not None:
            self._collection.delete_many({"file_id": {"$in": ids}})