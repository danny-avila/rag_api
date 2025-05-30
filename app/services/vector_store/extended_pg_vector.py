from typing import Optional

from sqlalchemy import delete
from sqlalchemy.orm import Session
from langchain_core.documents import Document
from langchain_community.vectorstores.pgvector import PGVector


class ExtendedPgVector(PGVector):
    def get_all_ids(self) -> list[str]:
        with Session(self._bind) as session:
            results = session.query(self.EmbeddingStore.custom_id).all()
            return [result[0] for result in results if result[0] is not None]

    def get_filtered_ids(self, ids: list[str]) -> list[str]:
        with Session(self._bind) as session:
            query = session.query(self.EmbeddingStore.custom_id).filter(
                self.EmbeddingStore.custom_id.in_(ids)
            )
            results = query.all()
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

    def get_remaining_chunks_for_file(
        self, file_id: str, excluding_ids: list[str]
    ) -> int:
        """Get count of remaining chunks for a file_id, excluding specified document IDs"""
        if not file_id:
            return 0

        with Session(self._bind) as session:
            from sqlalchemy import text

            # Get current collection to ensure we only count chunks from this collection
            collection = self.get_collection(session)
            if not collection:
                return 0

            if not excluding_ids:
                # No IDs to exclude - count all chunks for this file in this collection
                result = session.execute(
                    text(
                        """
                        SELECT COUNT(*) 
                        FROM langchain_pg_embedding 
                        WHERE cmetadata->>'file_id' = :file_id 
                        AND collection_id = :collection_id
                    """
                    ),
                    {"file_id": file_id, "collection_id": collection.uuid},
                ).scalar()
            else:
                # Count chunks excluding the specified IDs
                result = session.execute(
                    text(
                        """
                        SELECT COUNT(*) 
                        FROM langchain_pg_embedding 
                        WHERE cmetadata->>'file_id' = :file_id 
                        AND collection_id = :collection_id
                        AND custom_id NOT IN :excluding_ids
                    """
                    ),
                    {
                        "file_id": file_id,
                        "collection_id": collection.uuid,
                        "excluding_ids": tuple(excluding_ids),
                    },
                ).scalar()

            return result or 0

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
