from typing import Optional
from langchain_core.documents import Document
from langchain_core.runnables.config import run_in_executor
from .extended_pg_vector import ExtendedPgVector

class AsyncPgVector(ExtendedPgVector):
    async def get_all_ids(self) -> list[str]:
        return await run_in_executor(None, super().get_all_ids)
    
    async def get_filtered_ids(self, ids: list[str]) -> list[str]:
        return await run_in_executor(None, super().get_filtered_ids, ids)

    async def get_documents_by_ids(self, ids: list[str]) -> list[Document]:
        return await run_in_executor(None, super().get_documents_by_ids, ids)

    async def delete(
        self, ids: Optional[list[str]] = None, collection_only: bool = False
    ) -> None:
        await run_in_executor(None, self._delete_multiple, ids, collection_only)