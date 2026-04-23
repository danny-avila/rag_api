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
    """Raise if the POSTGRES_SCHEMA config won't let the app write to the
    target schema or read types from its fallbacks.

    schemas[0] is the write target — the schema pgvector's tables will land
    in — so the role needs USAGE + CREATE there. schemas[1:] are read-only
    search_path fallbacks for type/function resolution (e.g. an `extensions`
    schema that holds the `vector` type), so USAGE alone is sufficient; a
    role with CREATE here is fine too, but demanding it would reject the
    common least-privilege setup where writes are intentionally confined to
    the target schema.

    Silent fallback is worse than failing fast here: PostgreSQL resolves
    unqualified CREATE TABLE against the first schema in search_path where
    the role has CREATE privileges, so a typo or a missing grant on the
    target would land the pgvector tables in `public` (the entry appended
    by _build_search_path) instead of the intended namespace, silently
    defeating the isolation this feature is meant to provide.
    """
    from sqlalchemy import create_engine, text

    engine = create_engine(connection_string)
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT s.schema_name, "
                    "       has_schema_privilege(s.schema_name, 'USAGE') AS has_usage, "
                    "       has_schema_privilege(s.schema_name, 'CREATE') AS has_create "
                    "FROM information_schema.schemata s "
                    "WHERE s.schema_name = ANY(:names)"
                ),
                {"names": schemas},
            ).fetchall()
            found = {row[0]: (row[1], row[2]) for row in rows}

            target_schema = schemas[0]
            fallback_schemas = schemas[1:]

            missing = [s for s in schemas if s not in found]
            # USAGE is required on every listed schema (target to write,
            # fallbacks to look up types/functions at query time).
            no_usage = [s for s in schemas if s in found and not found[s][0]]
            # CREATE is only required on the target schema — the fallback
            # entries exist solely to make type resolution work.
            no_create_target = (
                [target_schema]
                if target_schema in found and not found[target_schema][1]
                else []
            )

            problems = []
            if missing:
                problems.append(f"does not exist: {missing!r}")
            if no_usage:
                problems.append(f"role lacks USAGE on: {no_usage!r}")
            if no_create_target:
                problems.append(f"role lacks CREATE on target: {no_create_target!r}")

            if problems:
                hint_target = (missing or no_usage or no_create_target)[0]
                raise ValueError(
                    "POSTGRES_SCHEMA: " + "; ".join(problems) + ". "
                    "Create/grant out-of-band first (e.g. "
                    f"`CREATE SCHEMA IF NOT EXISTS {hint_target}; "
                    f"GRANT USAGE, CREATE ON SCHEMA {hint_target} TO <app_user>;`)."
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
