import os
import time
import logging
from typing import Optional
from sqlalchemy import event
from sqlalchemy import delete
from sqlalchemy.orm import Session
from sqlalchemy.engine import Engine
from langchain_core.documents import Document
from langchain_community.vectorstores.pgvector import PGVector


class ExtendedPgVector(PGVector):
    _query_logging_setup = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_query_logging()

    def setup_query_logging(self):
        """Enable query logging for this vector store only if DEBUG_PGVECTOR_QUERIES is set"""
        # Only setup logging if the environment variable is set to a truthy value
        debug_queries = os.getenv("DEBUG_PGVECTOR_QUERIES", "").lower()
        if debug_queries not in ["true", "1", "yes", "on"]:
            return

        # Only setup once per class
        if ExtendedPgVector._query_logging_setup:
            return

        logger = logging.getLogger("pgvector.queries")
        logger.setLevel(logging.INFO)

        # Create handler if it doesn't exist
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - PGVECTOR QUERY - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        @event.listens_for(Engine, "before_cursor_execute")
        def receive_before_cursor_execute(
            conn, cursor, statement, parameters, context, executemany
        ):
            if "langchain_pg_embedding" in statement:
                context._query_start_time = time.time()
                logger.info(f"STARTING QUERY: {statement}")
                logger.info(f"PARAMETERS: {parameters}")

        @event.listens_for(Engine, "after_cursor_execute")
        def receive_after_cursor_execute(
            conn, cursor, statement, parameters, context, executemany
        ):
            if "langchain_pg_embedding" in statement:
                total = time.time() - context._query_start_time
                logger.info(f"COMPLETED QUERY in {total:.4f}s")
                logger.info("-" * 50)

        ExtendedPgVector._query_logging_setup = True

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
