import copy
import hashlib
from typing import Any, List, Optional, Sequence, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_mongodb import MongoDBAtlasVectorSearch


class AtlasMongoVector(MongoDBAtlasVectorSearch):
    @property
    def embedding_function(self) -> Embeddings:
        return self.embeddings

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs,
    ) -> List[str]:
        """Caller-supplied ``ids`` are intentionally ignored; IDs are derived from
        each document's content digest to ensure cross-batch uniqueness within a file.
        """
        if not documents:
            return []
        file_id = documents[0].metadata["file_id"]
        f_ids = [
            f"{file_id}_{doc.metadata.get('digest') or hashlib.md5(doc.page_content.encode()).hexdigest()}"
            for doc in documents
        ]
        return super().add_documents(documents, f_ids)

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

    @staticmethod
    def _owner_clause(owners: Sequence[str]) -> dict:
        return {"user_id": {"$in": list(dict.fromkeys(owners))}}

    def _file_scope_clause(self, ids: Sequence[str], owners: Sequence[str]) -> dict:
        """``(file id, owner)`` for the file-addressed routes.

        Scope is a required argument rather than an optional one: a
        caller-supplied file id is not an authorization.
        """
        return {
            "file_id": {"$in": list(dict.fromkeys(ids))},
            **self._owner_clause(owners),
        }

    def get_all_ids(self, owners: Sequence[str]) -> list[str]:
        """File ids this caller owns — never the deployment's whole set."""
        allowed = list(dict.fromkeys(owners))
        if not allowed:
            return []
        return self._collection.distinct("file_id", self._owner_clause(allowed))

    def get_filtered_ids(self, ids: Sequence[str], owners: Sequence[str]) -> list[str]:
        wanted = list(dict.fromkeys(ids))
        allowed = list(dict.fromkeys(owners))
        if not wanted or not allowed:
            return []
        return self._collection.distinct(
            "file_id", self._file_scope_clause(wanted, allowed)
        )

    def get_documents_by_ids(
        self, ids: Sequence[str], owners: Sequence[str]
    ) -> list[Document]:
        wanted = list(dict.fromkeys(ids))
        allowed = list(dict.fromkeys(owners))
        if not wanted or not allowed:
            return []
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
            for doc in self._collection.find(self._file_scope_clause(wanted, allowed))
        ]

    def delete_scoped(self, ids: Sequence[str], owners: Sequence[str]) -> None:
        """Delete the chunks of ``ids`` that ``owners`` actually own.

        A file id is caller-supplied and not unique across owners, so the delete
        carries the scope in its own predicate rather than trusting a separate
        existence check.
        """
        wanted = list(dict.fromkeys(ids))
        allowed = list(dict.fromkeys(owners))
        if not wanted or not allowed:
            return
        self._collection.delete_many(self._file_scope_clause(wanted, allowed))

    def delete(self, ids: Optional[list[str]] = None) -> None:
        # Delete documents by file_id
        if ids is not None:
            self._collection.delete_many({"file_id": {"$in": ids}})
