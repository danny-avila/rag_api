import asyncio
import time
from typing import Any, Optional

from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.documents import Document
from langchain_core.runnables.config import run_in_executor
from sqlalchemy.orm import Session


class ExtendedPgVector(PGVector):

    def get_all_ids(self) -> list[str]:
        time.sleep(5)

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


class AsnyPgVector(ExtendedPgVector):

    async def get_all_ids(self) -> list[str]:
        await asyncio.sleep(5)
        return await run_in_executor(None, super().get_all_ids)

    async def get_documents_by_ids(self, ids: list[str]) -> list[Document]:
        return await run_in_executor(None, super().get_documents_by_ids, ids)

    async def delete(
        self,
        ids: Optional[list[str]] = None,
        collection_only: bool = False,
        **kwargs: Any
    ) -> None:
        await run_in_executor(None, super().delete, ids, collection_only, **kwargs)
