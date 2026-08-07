import copy
import hashlib
from typing import Any, List, Optional, Sequence, Set, Tuple
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

    @staticmethod
    def _candidate_id_query(wanted: List[str]) -> dict:
        return {"$or": [{"_id": {"$in": wanted}}, {"digest": {"$in": wanted}}]}

    def probe_candidate_ids(
        self,
        ids: List[str],
        owners: List[str],
        tenants: Sequence[Optional[str]] = (None,),
    ) -> Tuple[Set[str], Set[str]]:
        """``(ids that exist at all, ids that exist within scope)`` — metadata only.

        Mirrors the pgvector probe so authorize-before-egress behaves identically
        on both backends.
        """
        wanted = list(dict.fromkeys(ids))
        if not wanted:
            return set(), set()

        allowed = set(dict.fromkeys(owners))
        tenant_values = set(tenants)
        cursor = self._collection.find(
            self._candidate_id_query(wanted),
            {"_id": 1, "digest": 1, "user_id": 1, "tenant_id": 1},
        ).sort("_id", 1)

        requested = set(wanted)
        existing: Set[str] = set()
        authorized: Set[str] = set()
        for doc in cursor:
            in_scope = (
                doc.get("user_id") in allowed and doc.get("tenant_id") in tenant_values
            )
            for key in (doc.get("_id"), doc.get("digest")):
                if key not in requested:
                    continue
                existing.add(key)
                if in_scope:
                    authorized.add(key)
        return existing, authorized

    def get_vectors_by_ids(
        self,
        ids: List[str],
        owners: List[str],
        tenants: Sequence[Optional[str]] = (None,),
    ) -> dict[str, List[float]]:
        """Stored chunk vectors for ``ids``, restricted to ``owners`` and ``tenants``.

        Mirrors the pgvector resolution: a candidate id matches the document
        ``_id`` or its chunk ``digest``, and scope is part of the query predicate
        rather than a post-filter. ``None`` in ``tenants`` matches missing
        fields, which is MongoDB's own ``$in: [null]`` semantic.
        """
        wanted = list(dict.fromkeys(ids))
        allowed = list(dict.fromkeys(owners))
        if not wanted or not allowed or not tenants:
            return {}

        embedding_key = getattr(self, "_embedding_key", "embedding")
        cursor = self._collection.find(
            {
                "$and": [
                    self._candidate_id_query(wanted),
                    {"user_id": {"$in": allowed}},
                    {"tenant_id": {"$in": list(tenants)}},
                ]
            },
            {"_id": 1, "digest": 1, embedding_key: 1},
        ).sort("_id", 1)

        requested = set(wanted)
        vectors: dict[str, List[float]] = {}
        for doc in cursor:
            embedding = doc.get(embedding_key)
            if not embedding:
                continue
            vector = [float(component) for component in embedding]
            for key in (doc.get("_id"), doc.get("digest")):
                if key in requested and key not in vectors:
                    vectors[key] = vector
        return vectors

    def delete(self, ids: Optional[list[str]] = None) -> None:
        # Delete documents by file_id
        if ids is not None:
            self._collection.delete_many({"file_id": {"$in": ids}})
