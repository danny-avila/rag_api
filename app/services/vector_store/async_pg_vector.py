from typing import Callable, Optional, List, Tuple, Dict, Any, TypeVar
import asyncio
from concurrent.futures import Executor
from langchain_core.documents import Document
from .extended_pg_vector import ExtendedPgVector

T = TypeVar("T")


class AsyncPgVector(ExtendedPgVector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._thread_pool = None

    def _get_thread_pool(self):
        if self._thread_pool is None:
            try:
                loop = asyncio.get_running_loop()
                self._thread_pool = getattr(loop, "_default_executor", None)
            except Exception:
                pass
        return self._thread_pool

    @staticmethod
    async def _run_in_executor(
        executor: Executor | None,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Run a sync callable in a thread pool executor.

        Wraps the call to convert StopIteration into RuntimeError.
        StopIteration cannot be set on an asyncio.Future — it raises
        TypeError and leaves the Future pending forever.
        """

        def wrapper() -> T:
            try:
                return func(*args, **kwargs)
            except StopIteration as exc:
                raise RuntimeError from exc

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, wrapper)

    async def get_all_ids(self, executor=None) -> list[str]:
        executor = executor or self._get_thread_pool()
        return await self._run_in_executor(executor, super().get_all_ids)

    async def get_filtered_ids(self, ids: list[str], executor=None) -> list[str]:
        executor = executor or self._get_thread_pool()
        return await self._run_in_executor(executor, super().get_filtered_ids, ids)

    async def get_documents_by_ids(
        self, ids: list[str], executor=None
    ) -> list[Document]:
        executor = executor or self._get_thread_pool()
        return await self._run_in_executor(executor, super().get_documents_by_ids, ids)

    async def delete(
        self,
        ids: Optional[list[str]] = None,
        collection_only: bool = False,
        executor=None,
    ) -> None:
        executor = executor or self._get_thread_pool()
        await self._run_in_executor(
            executor, self._delete_multiple, ids, collection_only
        )

    async def asimilarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        executor=None,
    ) -> List[Tuple[Document, float]]:
        """Async version of similarity_search_with_score_by_vector"""
        executor = executor or self._get_thread_pool()
        return await self._run_in_executor(
            executor,
            super().similarity_search_with_score_by_vector,
            embedding,
            k,
            filter,
        )

    async def aadd_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        executor=None,
        **kwargs,
    ) -> List[str]:
        """Async version of add_documents"""
        executor = executor or self._get_thread_pool()
        return await self._run_in_executor(
            executor, super().add_documents, documents, ids=ids, **kwargs
        )
