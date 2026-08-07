import os
import time
import logging
from typing import Optional, Any, Dict, List, Sequence, Set, Tuple, Union
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
        elif operator == "$in" and any(item is None for item in filter_value):
            # MongoDB's `$in: [null]` matches documents where the field is null
            # *or absent*. Scope predicates depend on that to treat chunks
            # written before tenants were recorded as base-tenant content, so
            # the same semantic is implemented here rather than dropping the
            # null and silently narrowing the filter.
            column = self.EmbeddingStore.cmetadata[field].astext
            present = [str(item) for item in filter_value if item is not None]
            if not present:
                return column.is_(None)
            return sqlalchemy.or_(column.in_(present), column.is_(None))

        return super()._handle_field_filter(field, value)

    def get_all_ids(
        self, owners: Sequence[str], tenants: Sequence[Optional[str]]
    ) -> list[str]:
        """File ids this caller's scope holds — never the deployment's whole set."""
        allowed = list(dict.fromkeys(owners))
        if not allowed or not tenants:
            return []
        with Session(self._bind) as session:
            collection_clause = self._collection_clause(session)
            if collection_clause is None:
                return []
            results = (
                session.query(self.EmbeddingStore.custom_id)
                .filter(collection_clause)
                .filter(self._scope_clause(allowed, tenants))
                .distinct()
                .all()
            )
            return [result[0] for result in results if result[0] is not None]

    def get_filtered_ids(
        self,
        ids: Sequence[str],
        owners: Sequence[str],
        tenants: Sequence[Optional[str]],
    ) -> list[str]:
        """Which of ``ids`` exist inside ``owners`` and ``tenants``.

        Scope is a required argument rather than an optional one: a
        caller-supplied file id is not an authorization, and an existence answer
        computed without the owner and tenant predicate is an oracle over every
        file in the deployment.
        """
        wanted = list(dict.fromkeys(ids))
        allowed = list(dict.fromkeys(owners))
        if not wanted or not allowed or not tenants:
            return []
        with Session(self._bind) as session:
            file_clause = self._file_scope_clause(session, wanted, allowed, tenants)
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
        self,
        ids: Sequence[str],
        owners: Sequence[str],
        tenants: Sequence[Optional[str]],
    ) -> list[Document]:
        """Chunks of ``ids`` owned inside ``owners`` and ``tenants``.

        The scope goes into the SQL predicate, so a foreign chunk is never read
        into this process rather than being filtered out after the fact.
        """
        wanted = list(dict.fromkeys(ids))
        allowed = list(dict.fromkeys(owners))
        if not wanted or not allowed or not tenants:
            return []
        with Session(self._bind) as session:
            file_clause = self._file_scope_clause(session, wanted, allowed, tenants)
            if file_clause is None:
                return []
            results = session.query(self.EmbeddingStore).filter(file_clause).all()
            return [
                Document(page_content=result.document, metadata=result.cmetadata or {})
                for result in results
            ]

    def delete_scoped(
        self,
        ids: Sequence[str],
        owners: Sequence[str],
        tenants: Sequence[Optional[str]],
    ) -> None:
        """Delete the chunks of ``ids`` that ``owners``/``tenants`` actually own.

        A file id is caller-supplied and not unique across owners — anyone may
        upload under a chosen ``file_id`` — so the DELETE carries the scope in
        its own predicate rather than trusting a separate existence check.
        """
        self._delete_scoped(ids, owners, tenants)

    def _delete_scoped(
        self,
        ids: Sequence[str],
        owners: Sequence[str],
        tenants: Sequence[Optional[str]],
    ) -> None:
        wanted = list(dict.fromkeys(ids))
        allowed = list(dict.fromkeys(owners))
        if not wanted or not allowed or not tenants:
            return
        with Session(self._bind) as session:
            file_clause = self._file_scope_clause(session, wanted, allowed, tenants)
            if file_clause is None:
                return
            session.execute(delete(self.EmbeddingStore).where(file_clause))
            session.commit()

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

    def _candidate_id_clause(self, wanted: List[str]):
        """Match a caller's candidate id against the two per-chunk handles.

        ``custom_id`` is the file id and is shared by every chunk of a file, so
        it identifies no single vector; the row ``uuid`` and the chunk
        ``digest`` recorded in metadata are the handles a caller can hold.
        """
        return sqlalchemy.or_(
            sqlalchemy.cast(self.EmbeddingStore.uuid, sqlalchemy.String).in_(wanted),
            self.EmbeddingStore.cmetadata["digest"].astext.in_(wanted),
        )

    def _collection_clause(self, session: Session):
        """Restrict a query to the collection this store serves, or ``None``.

        ``langchain_pg_embedding`` is shared by every collection in the
        database, so a candidate lookup that omits ``collection_id`` can see —
        and reuse the vector of — a row this store does not serve. ``None`` means
        the collection has not been created yet, which is indistinguishable from
        it holding no rows.
        """
        collection = self.get_collection(session)
        if collection is None:
            return None
        return self.EmbeddingStore.collection_id == collection.uuid

    def _scope_clause(self, owners: List[str], tenants: Sequence[Optional[str]]):
        owner_column = self.EmbeddingStore.cmetadata["user_id"].astext
        tenant_column = self.EmbeddingStore.cmetadata["tenant_id"].astext
        present = [str(tenant) for tenant in tenants if tenant is not None]
        if any(tenant is None for tenant in tenants):
            tenant_clause = (
                sqlalchemy.or_(tenant_column.in_(present), tenant_column.is_(None))
                if present
                else tenant_column.is_(None)
            )
        else:
            tenant_clause = tenant_column.in_(present)
        return sqlalchemy.and_(owner_column.in_(owners), tenant_clause)

    def _file_scope_clause(
        self,
        session: Session,
        ids: Sequence[str],
        owners: Sequence[str],
        tenants: Sequence[Optional[str]],
    ):
        """``(collection, file id, owner, tenant)`` for the file-addressed routes.

        ``None`` means this store's collection does not exist yet, which is
        indistinguishable from it holding no rows.
        """
        collection_clause = self._collection_clause(session)
        if collection_clause is None:
            return None
        return sqlalchemy.and_(
            collection_clause,
            self.EmbeddingStore.custom_id.in_(list(ids)),
            self._scope_clause(list(owners), tenants),
        )

    def probe_candidate_ids(
        self,
        ids: Sequence[str],
        owners: Sequence[str],
        tenants: Sequence[Optional[str]],
    ) -> Tuple[Set[str], Set[str]]:
        """``(ids that exist at all, ids that exist within scope)``.

        Metadata only — no vectors, no document text. This is the
        authorize-before-egress check: a candidate id that exists in the store
        but resolves to nothing inside the caller's scope must be refused
        *before* its caller-supplied text is sent to the inference gateway.

        "The store" means this store's collection. A row belonging to another
        collection in the same database is not served here at all, so counting
        it as existing would only manufacture a 403 for a candidate this
        deployment never held.
        """
        wanted = list(dict.fromkeys(ids))
        if not wanted:
            return set(), set()

        allowed = list(dict.fromkeys(owners))
        with Session(self._bind) as session:
            collection_clause = self._collection_clause(session)
            if collection_clause is None:
                return set(), set()
            rows = (
                session.query(
                    self.EmbeddingStore.uuid,
                    self.EmbeddingStore.cmetadata["digest"].astext,
                    self.EmbeddingStore.cmetadata["user_id"].astext,
                    self.EmbeddingStore.cmetadata["tenant_id"].astext,
                )
                .filter(collection_clause)
                .filter(self._candidate_id_clause(wanted))
                .order_by(self.EmbeddingStore.uuid)
                .all()
            )

        requested = set(wanted)
        tenant_values = set(tenants)
        existing: Set[str] = set()
        authorized: Set[str] = set()
        for row_uuid, row_digest, row_owner, row_tenant in rows:
            in_scope = row_owner in allowed and row_tenant in tenant_values
            for key in (str(row_uuid), row_digest):
                if key not in requested:
                    continue
                existing.add(key)
                if in_scope:
                    authorized.add(key)
        return existing, authorized

    def get_vectors_by_ids(
        self,
        ids: Sequence[str],
        owners: Sequence[str],
        tenants: Sequence[Optional[str]] = (None,),
    ) -> Dict[str, List[float]]:
        """Stored chunk vectors for ``ids``, restricted to ``owners`` and ``tenants``.

        Scope is part of the SQL predicate: a foreign chunk is never fetched, so
        it can never be scored, counted, or read into this process. Collection
        is part of it too, so a matching digest in a sibling collection cannot
        have its vector reused here.
        """
        wanted = list(dict.fromkeys(ids))
        allowed = list(dict.fromkeys(owners))
        if not wanted or not allowed or not tenants:
            return {}

        with Session(self._bind) as session:
            collection_clause = self._collection_clause(session)
            if collection_clause is None:
                return {}
            rows = (
                session.query(
                    self.EmbeddingStore.uuid,
                    self.EmbeddingStore.cmetadata,
                    self.EmbeddingStore.embedding,
                )
                .filter(collection_clause)
                .filter(self._candidate_id_clause(wanted))
                .filter(self._scope_clause(allowed, tenants))
                .order_by(self.EmbeddingStore.uuid)
                .all()
            )

        requested = set(wanted)
        vectors: Dict[str, List[float]] = {}
        for row_uuid, metadata, embedding in rows:
            if embedding is None:
                continue
            vector = [float(component) for component in embedding]
            for key in (str(row_uuid), (metadata or {}).get("digest")):
                if key in requested and key not in vectors:
                    vectors[key] = vector
        return vectors

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
