import logging
from typing import Any, List, Optional

from pymongo import MongoClient
from langchain_core.embeddings import Embeddings

from .async_pg_vector import AsyncPgVector
from .atlas_mongo_vector import AtlasMongoVector
from .extended_pg_vector import ExtendedPgVector

logger = logging.getLogger(__name__)

# Holds the MongoClient so it can be closed on shutdown.
_mongo_client: Optional[MongoClient] = None


def _parse_schemas(schema: str) -> List[str]:
    """Split POSTGRES_SCHEMA's comma-separated value into a clean list."""
    return [s.strip() for s in schema.split(",") if s.strip()]


def _build_search_path(schemas: List[str]) -> str:
    """Build a Postgres search_path value that includes every requested
    schema plus `public` (appended if missing). pgvector installs the
    `vector` data type into whatever schema its CREATE EXTENSION targets
    (almost always `public`), and unqualified type names are resolved
    against search_path — so bare `search_path=myapp` breaks
    `CREATE TABLE ... vector(...)` with `type "vector" does not exist`.
    """
    parts = list(schemas)
    if "public" not in parts:
        parts.append("public")
    return ",".join(parts)


def _verify_schemas_exist(connection_string: str, schemas: List[str]) -> None:
    """Raise if any requested schema doesn't exist in the target database.

    Silent fallback is worse than failing fast here: PostgreSQL resolves
    unqualified CREATE TABLE against the first schema in search_path where
    the role has CREATE privileges, so a typo in POSTGRES_SCHEMA would
    land the pgvector tables in `public` instead of the intended namespace,
    silently defeating the isolation this feature is meant to provide.
    """
    from sqlalchemy import create_engine, text

    engine = create_engine(connection_string)
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT schema_name FROM information_schema.schemata "
                    "WHERE schema_name = ANY(:names)"
                ),
                {"names": schemas},
            ).fetchall()
            existing = {r[0] for r in rows}
            missing = [s for s in schemas if s not in existing]
            if missing:
                raise ValueError(
                    f"POSTGRES_SCHEMA: schema(s) {missing!r} do not exist. "
                    "Create them out-of-band first (e.g. "
                    f"`CREATE SCHEMA IF NOT EXISTS {missing[0]}; "
                    f"GRANT USAGE, CREATE ON SCHEMA {missing[0]} TO <app_user>;`)."
                )
    finally:
        engine.dispose()


def get_vector_store(
    connection_string: str,
    embeddings: Embeddings,
    collection_name: str,
    mode: str = "sync",
    search_index: Optional[str] = None,
    create_extension: bool = True,
    pool_pre_ping: bool = True,
    pool_recycle: int = -1,
    schema: Optional[str] = None,
):
    """Create a vector store instance for the given mode.

    Note: For 'atlas-mongo' mode, the MongoClient is stored at module level
    so it can be closed on shutdown via close_vector_store_connections().

    Set create_extension=False when the Postgres user lacks superuser
    privileges and the `vector` extension is already installed out-of-band
    (e.g. managed Postgres services like RDS, Azure Database for PostgreSQL).

    pool_pre_ping issues a SELECT 1 before handing out a pooled connection,
    so stale/dead connections (e.g. dropped by a remote server or middlebox
    idle timeout) are detected and replaced instead of surfacing as a query
    error. pool_recycle<=0 disables periodic recycling (SQLAlchemy default);
    set a positive number of seconds when the server enforces an idle or
    max-lifetime limit.

    Set schema to prepend that schema to every connection's search_path so
    langchain's pgvector tables are created and queried there — use this to
    keep the vector store logically separated when sharing a database with
    other services. The schema must already exist (fails fast via
    information_schema.schemata if missing). Multiple schemas may be
    supplied as a comma-separated list; `public` is always appended so the
    `vector` data type stays resolvable when the extension was created
    there.
    """
    global _mongo_client

    engine_args: dict = {"pool_pre_ping": pool_pre_ping}
    if pool_recycle > 0:
        engine_args["pool_recycle"] = pool_recycle
    if schema:
        schemas = _parse_schemas(schema)
        if schemas:
            _verify_schemas_exist(connection_string, schemas)
            search_path = _build_search_path(schemas)
            engine_args["connect_args"] = {"options": f"-csearch_path={search_path}"}

    if mode == "sync":
        return ExtendedPgVector(
            connection_string=connection_string,
            embedding_function=embeddings,
            collection_name=collection_name,
            use_jsonb=True,
            create_extension=create_extension,
            engine_args=engine_args,
        )
    elif mode == "async":
        return AsyncPgVector(
            connection_string=connection_string,
            embedding_function=embeddings,
            collection_name=collection_name,
            use_jsonb=True,
            create_extension=create_extension,
            engine_args=engine_args,
        )
    elif mode == "atlas-mongo":
        if _mongo_client is not None:
            _mongo_client.close()
        _mongo_client = MongoClient(connection_string)
        mongo_db = _mongo_client.get_database()
        mong_collection = mongo_db[collection_name]
        return AtlasMongoVector(
            collection=mong_collection, embedding=embeddings, index_name=search_index
        )
    else:
        raise ValueError(
            "Invalid mode specified. Choose 'sync', 'async', or 'atlas-mongo'."
        )


def close_vector_store_connections(vector_store: Any) -> None:
    """Close connections held by the vector store and its backing clients.

    Closes the module-level MongoClient (if atlas-mongo mode was used) and
    disposes the SQLAlchemy engine on the vector store (if pgvector mode).
    Safe to call multiple times.
    """
    global _mongo_client

    # Close MongoDB client if one was created
    if _mongo_client is not None:
        try:
            _mongo_client.close()
            logger.info("MongoDB client closed")
        except Exception as e:
            logger.warning("Failed to close MongoDB client: %s", e)
        finally:
            _mongo_client = None

    # Dispose SQLAlchemy engine if the vector store has one
    engine = getattr(vector_store, "_bind", None)
    if engine is not None and hasattr(engine, "dispose"):
        try:
            engine.dispose()
            logger.info("SQLAlchemy engine disposed")
        except Exception as e:
            logger.warning("Failed to dispose SQLAlchemy engine: %s", e)
