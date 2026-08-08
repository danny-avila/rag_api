import os
import time
import logging
from typing import Optional, Any, Dict, List, Sequence, Union
import sqlalchemy
from sqlalchemy import event
from sqlalchemy import delete
from sqlalchemy.orm import Session
from sqlalchemy.engine import Engine
from langchain_core.documents import Document
from langchain_community.vectorstores.pgvector import (
    PGVector,
    COMPARISONS_TO_NATIVE,
    SUPPORTED_OPERATORS,
)


class ExtendedPgVector(PGVector):
    _query_logging_setup = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_query_logging()

    @staticmethod
    def _sanitize_parameters_for_logging(
        parameters: Union[Dict, List, tuple, Any]
    ) -> Any:
        """Sanitize parameters for logging by truncating embeddings and large values."""
        if parameters is None:
            return parameters

        if isinstance(parameters, dict):
            sanitized = {}
            for key, value in parameters.items():
                # Check if the key contains 'embedding' or if the value looks like an embedding vector
                if "embedding" in str(key).lower() or (
                    isinstance(value, (list, tuple))
                    and len(value) > 10
                    and all(isinstance(x, (int, float)) for x in value[:10])
                ):
                    sanitized[key] = f"<embedding vector of length {len(value)}>"
                elif isinstance(value, str) and len(value) > 500:
                    sanitized[key] = value[:500] + "... (truncated)"
                elif isinstance(value, (dict, list, tuple)):
                    sanitized[key] = ExtendedPgVector._sanitize_parameters_for_logging(
                        value
                    )
                else:
                    sanitized[key] = value
            return sanitized
        elif isinstance(parameters, (list, tuple)):
            sanitized = []
            # Check if this is a list of embeddings
            if len(parameters) > 0 and all(
                isinstance(item, (list, tuple))
                and len(item) > 10
                and all(isinstance(x, (int, float)) for x in item[: min(10, len(item))])
                for item in parameters
            ):
                return f"<{len(parameters)} embedding vectors>"

            for item in parameters:
                if (
                    isinstance(item, (list, tuple))
                    and len(item) > 10
                    and all(isinstance(x, (int, float)) for x in item[:10])
                ):
                    sanitized.append(f"<embedding vector of length {len(item)}>")
                elif isinstance(item, str) and len(item) > 500:
                    sanitized.append(item[:500] + "... (truncated)")
                elif isinstance(item, (dict, list, tuple)):
                    sanitized.append(
                        ExtendedPgVector._sanitize_parameters_for_logging(item)
                    )
                else:
                    sanitized.append(item)
            return type(parameters)(sanitized)
        else:
            return parameters

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
                sanitized_params = ExtendedPgVector._sanitize_parameters_for_logging(
                    parameters
                )
                logger.info(f"PARAMETERS: {sanitized_params}")

        @event.listens_for(Engine, "after_cursor_execute")
        def receive_after_cursor_execute(
            conn, cursor, statement, parameters, context, executemany
        ):
            if "langchain_pg_embedding" in statement:
                total = time.time() - context._query_start_time
                logger.info(f"COMPLETED QUERY in {total:.4f}s")
                logger.info("-" * 50)

        ExtendedPgVector._query_logging_setup = True

    def _handle_field_filter(self, field: str, value: Any) -> Any:
        """Override LangChain's filter to avoid jsonb_path_match() for equality ops.

        LangChain's default _handle_field_filter uses func.jsonb_path_match() for
        $eq/$ne/$lt/$gt etc. That function-call predicate cannot use B-tree expression
        indexes like (cmetadata->>'file_id') or GIN jsonb_path_ops indexes, forcing
        PostgreSQL into sequential scans on large tables.

        This override rewrites $eq and $ne to use the ->>' astext operator instead,
        producing WHERE (cmetadata->>'field') = 'value' which hits expression indexes.
        All other operators ($lt, $gt, $in, $between, etc.) delegate to the parent.
        """
        if not isinstance(field, str):
            raise ValueError(
                f"field should be a string but got: {type(field)} with value: {field}"
            )
        if field.startswith("$"):
            raise ValueError(
                f"Invalid filter condition. Expected a field but got an operator: {field}"
            )
        if not field.isidentifier():
            raise ValueError(
                f"Invalid field name: {field}. Expected a valid identifier."
            )

        if isinstance(value, dict):
            if len(value) != 1:
                raise ValueError(
                    "Invalid filter condition. Expected a value which "
                    "is a dictionary with a single key that corresponds to an operator "
                    f"but got a dictionary with {len(value)} keys. The first few "
                    f"keys are: {list(value.keys())[:3]}"
                )
            operator, filter_value = list(value.items())[0]
            if operator not in SUPPORTED_OPERATORS:
                raise ValueError(
                    f"Invalid operator: {operator}. "
                    f"Expected one of {SUPPORTED_OPERATORS}"
                )
        else:
            operator = "$eq"
            filter_value = value

        if operator == "$eq":
            return self.EmbeddingStore.cmetadata[field].astext == str(filter_value)
        elif operator == "$ne":
            return self.EmbeddingStore.cmetadata[field].astext != str(filter_value)

        return super()._handle_field_filter(field, value)

    def _collection_clause(self, session: Session):
        """Restrict a query to the collection this store serves, or ``None``.

        ``langchain_pg_embedding`` is shared by every collection in the
        database, so a lookup that omits ``collection_id`` can read a row this
        store does not serve. ``None`` means the collection has not been created
        yet, which is indistinguishable from it holding no rows.
        """
        collection = self.get_collection(session)
        if collection is None:
            return None
        return self.EmbeddingStore.collection_id == collection.uuid

    def _owner_clause(self, owners: Sequence[str]):
        return self.EmbeddingStore.cmetadata["user_id"].astext.in_(list(owners))

    def _file_scope_clause(
        self, session: Session, ids: Sequence[str], owners: Sequence[str]
    ):
        """``(collection, file id, owner)`` for the file-addressed routes.

        ``None`` means this store's collection does not exist yet, which is
        indistinguishable from it holding no rows.
        """
        collection_clause = self._collection_clause(session)
        if collection_clause is None:
            return None
        return sqlalchemy.and_(
            collection_clause,
            self.EmbeddingStore.custom_id.in_(list(ids)),
            self._owner_clause(owners),
        )

    def get_all_ids(self, owners: Sequence[str]) -> list[str]:
        """File ids this caller owns — never the deployment's whole set."""
        allowed = list(dict.fromkeys(owners))
        if not allowed:
            return []
        with Session(self._bind) as session:
            collection_clause = self._collection_clause(session)
            if collection_clause is None:
                return []
            results = (
                session.query(self.EmbeddingStore.custom_id)
                .filter(collection_clause)
                .filter(self._owner_clause(allowed))
                .distinct()
                .all()
            )
            return [result[0] for result in results if result[0] is not None]

    def get_filtered_ids(self, ids: Sequence[str], owners: Sequence[str]) -> list[str]:
        """Which of ``ids`` exist inside ``owners``.

        Scope is a required argument rather than an optional one: a
        caller-supplied file id is not an authorization, and an existence answer
        computed without the owner predicate is an oracle over every file in the
        deployment.
        """
        wanted = list(dict.fromkeys(ids))
        allowed = list(dict.fromkeys(owners))
        if not wanted or not allowed:
            return []
        with Session(self._bind) as session:
            file_clause = self._file_scope_clause(session, wanted, allowed)
            if file_clause is None:
                return []
            results = (
                session.query(self.EmbeddingStore.custom_id)
                .filter(file_clause)
                .distinct()
                .all()
            )
            return [result[0] for result in results if result[0] is not None]

    def get_documents_by_ids(
        self, ids: Sequence[str], owners: Sequence[str]
    ) -> list[Document]:
        """Chunks of ``ids`` owned by ``owners``.

        The scope goes into the SQL predicate, so a foreign chunk is never read
        into this process rather than being filtered out after the fact.
        """
        wanted = list(dict.fromkeys(ids))
        allowed = list(dict.fromkeys(owners))
        if not wanted or not allowed:
            return []
        with Session(self._bind) as session:
            file_clause = self._file_scope_clause(session, wanted, allowed)
            if file_clause is None:
                return []
            results = session.query(self.EmbeddingStore).filter(file_clause).all()
            return [
                Document(page_content=result.document, metadata=result.cmetadata or {})
                for result in results
            ]

    def _delete_by_metadata(self, metadata_filter: Dict[str, Any]) -> None:
        """Delete rows in this collection that exactly match metadata values."""
        if not metadata_filter:
            raise ValueError("metadata_filter must not be empty")

        with Session(self._bind) as session:
            collection = self.get_collection(session)
            if not collection:
                self.logger.warning("Collection not found")
                return

            stmt = delete(self.EmbeddingStore).where(
                self.EmbeddingStore.collection_id == collection.uuid
            )
            for field, value in metadata_filter.items():
                stmt = stmt.where(self._handle_field_filter(field, value))

            session.execute(stmt)
            session.commit()

    def delete_scoped(self, ids: Sequence[str], owners: Sequence[str]) -> None:
        """Delete the chunks of ``ids`` that ``owners`` actually own.

        A file id is caller-supplied and not unique across owners — anyone may
        upload under a chosen ``file_id`` — so the DELETE carries the scope in
        its own predicate rather than trusting a separate existence check.
        """
        wanted = list(dict.fromkeys(ids))
        allowed = list(dict.fromkeys(owners))
        if not wanted or not allowed:
            return
        with Session(self._bind) as session:
            file_clause = self._file_scope_clause(session, wanted, allowed)
            if file_clause is None:
                return
            session.execute(delete(self.EmbeddingStore).where(file_clause))
            session.commit()

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
