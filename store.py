from typing import Any, Optional
from sqlalchemy import delete
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.documents import Document
from langchain_core.runnables.config import run_in_executor
from sqlalchemy.orm import Session
from langchain_community.vectorstores import Qdrant
import qdrant_client as client
from qdrant_client.http import models 

class ExtendedQdrant(Qdrant):
    def delete_vectors_by_source_document(self, source_document_ids: list[str]) -> None:
        """Delete vectors from the collection associated with specific source documents.

        Args:
            source_document_ids: The IDs of the source documents whose associated vectors should be deleted.
        """
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
            results = client.scroll(
                collection_name="{collection_name}",
                scroll_filter=models.Filter(
                    must_not=[
                    models.FieldCondition(
                        key="metadata.file_id",
                        match=models.MatchAny(any="source_document_ids"),
                    ),
                ],
                ),
            )
            return [result[0] for result in results if result[0] is not None]

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
        self,
        ids: Optional[list[str]] = None,
        collection_only: bool = False
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
            
class AsyncQdrant(ExtendedQdrant):
    async def get_all_ids(self) -> list[str]:
        return await run_in_executor(None, super().get_all_ids)

    async def get_documents_by_ids(self, ids: list[str]) -> list[Document]:
        return await run_in_executor(None, super().get_documents_by_ids, ids)

    async def delete_vectors(
        self,
        ids: Optional[list[str]] = None
    ) -> None:
        # Garantir que o argumento correto está sendo passado
        await run_in_executor(None, self.delete_vectors_by_source_document, ids)

 
    