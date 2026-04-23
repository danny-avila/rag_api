"""Tests for vector store factory shutdown and cleanup logic."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from app.services.vector_store import factory
from app.services.vector_store.factory import close_vector_store_connections


def test_close_vector_store_connections_mongo():
    """close_vector_store_connections closes the module-level MongoClient."""
    mock_client = MagicMock()
    factory._mongo_client = mock_client

    try:
        close_vector_store_connections(vector_store=None)
        mock_client.close.assert_called_once()
        assert factory._mongo_client is None
    finally:
        factory._mongo_client = None


def test_close_vector_store_connections_sqlalchemy():
    """close_vector_store_connections disposes the SQLAlchemy engine on the vector store."""
    mock_engine = MagicMock()
    mock_engine.dispose = MagicMock()

    mock_vs = MagicMock()
    mock_vs._bind = mock_engine

    close_vector_store_connections(mock_vs)
    mock_engine.dispose.assert_called_once()


def test_close_vector_store_connections_idempotent():
    """Calling close_vector_store_connections twice is safe."""
    mock_client = MagicMock()
    factory._mongo_client = mock_client

    mock_engine = MagicMock()
    mock_vs = MagicMock()
    mock_vs._bind = mock_engine

    try:
        close_vector_store_connections(mock_vs)
        close_vector_store_connections(mock_vs)

        # Mongo closed once, then global set to None so second call skips it
        mock_client.close.assert_called_once()
        # Engine dispose called twice (harmless — SQLAlchemy handles it)
        assert mock_engine.dispose.call_count == 2
    finally:
        factory._mongo_client = None


def test_close_vector_store_connections_no_bind():
    """close_vector_store_connections handles vector stores without _bind."""
    mock_vs = MagicMock(spec=[])  # No attributes at all
    # Should not raise
    close_vector_store_connections(mock_vs)


def test_close_vector_store_connections_none():
    """close_vector_store_connections handles None vector store."""
    close_vector_store_connections(None)


def test_get_vector_store_atlas_mongo_closes_previous_client():
    """Calling get_vector_store(atlas-mongo) twice closes the first MongoClient."""
    factory._mongo_client = None

    with patch("app.services.vector_store.factory.MongoClient") as MockMC:
        mock_client_1 = MagicMock()
        mock_client_2 = MagicMock()
        MockMC.side_effect = [mock_client_1, mock_client_2]

        mock_embeddings = MagicMock()

        with patch("app.services.vector_store.factory.AtlasMongoVector"):
            factory.get_vector_store(
                "conn1", mock_embeddings, "coll", mode="atlas-mongo", search_index="idx"
            )
            assert factory._mongo_client is mock_client_1
            mock_client_1.close.assert_not_called()

            factory.get_vector_store(
                "conn2", mock_embeddings, "coll", mode="atlas-mongo", search_index="idx"
            )
            # First client should have been closed before overwrite
            mock_client_1.close.assert_called_once()
            assert factory._mongo_client is mock_client_2

    factory._mongo_client = None


def test_get_vector_store_sync_passes_use_jsonb():
    """Sync PgVector must be instantiated with use_jsonb=True."""
    with patch("app.services.vector_store.factory.ExtendedPgVector") as MockPG:
        mock_embeddings = MagicMock()
        factory.get_vector_store("conn", mock_embeddings, "coll", mode="sync")
        _, kwargs = MockPG.call_args
        assert kwargs.get("use_jsonb") is True


def test_get_vector_store_async_passes_use_jsonb():
    """Async PgVector must be instantiated with use_jsonb=True."""
    with patch("app.services.vector_store.factory.AsyncPgVector") as MockPG:
        mock_embeddings = MagicMock()
        factory.get_vector_store("conn", mock_embeddings, "coll", mode="async")
        _, kwargs = MockPG.call_args
        assert kwargs.get("use_jsonb") is True


def test_get_vector_store_defaults_create_extension_true():
    """PgVector must default to create_extension=True for back-compat."""
    with patch("app.services.vector_store.factory.AsyncPgVector") as MockPG:
        mock_embeddings = MagicMock()
        factory.get_vector_store("conn", mock_embeddings, "coll", mode="async")
        _, kwargs = MockPG.call_args
        assert kwargs.get("create_extension") is True


def test_get_vector_store_propagates_create_extension_false():
    """create_extension=False must reach the underlying PGVector — the escape
    hatch for managed Postgres where the app user can't CREATE EXTENSION."""
    with patch("app.services.vector_store.factory.AsyncPgVector") as MockPG:
        mock_embeddings = MagicMock()
        factory.get_vector_store(
            "conn", mock_embeddings, "coll", mode="async", create_extension=False
        )
        _, kwargs = MockPG.call_args
        assert kwargs.get("create_extension") is False

    with patch("app.services.vector_store.factory.ExtendedPgVector") as MockPG:
        mock_embeddings = MagicMock()
        factory.get_vector_store(
            "conn", mock_embeddings, "coll", mode="sync", create_extension=False
        )
        _, kwargs = MockPG.call_args
        assert kwargs.get("create_extension") is False


def test_get_vector_store_defaults_enable_pool_pre_ping():
    """Default engine_args must enable pool_pre_ping — the back-compat-safe
    default that prevents dead-connection errors on remote Postgres."""
    with patch("app.services.vector_store.factory.AsyncPgVector") as MockPG:
        mock_embeddings = MagicMock()
        factory.get_vector_store("conn", mock_embeddings, "coll", mode="async")
        _, kwargs = MockPG.call_args
        engine_args = kwargs.get("engine_args")
        assert engine_args == {"pool_pre_ping": True}


def test_get_vector_store_can_disable_pool_pre_ping():
    """pool_pre_ping=False must be propagated — the escape hatch for callers
    that want to avoid the per-checkout SELECT 1 overhead."""
    with patch("app.services.vector_store.factory.AsyncPgVector") as MockPG:
        mock_embeddings = MagicMock()
        factory.get_vector_store(
            "conn", mock_embeddings, "coll", mode="async", pool_pre_ping=False
        )
        _, kwargs = MockPG.call_args
        assert kwargs.get("engine_args") == {"pool_pre_ping": False}


def test_get_vector_store_propagates_pool_recycle():
    """pool_recycle>0 must appear in engine_args; pool_recycle<=0 must be
    omitted so SQLAlchemy falls back to its own default (no recycling)."""
    with patch("app.services.vector_store.factory.AsyncPgVector") as MockPG:
        mock_embeddings = MagicMock()
        factory.get_vector_store(
            "conn", mock_embeddings, "coll", mode="async", pool_recycle=1800
        )
        _, kwargs = MockPG.call_args
        engine_args = kwargs.get("engine_args")
        assert engine_args == {"pool_pre_ping": True, "pool_recycle": 1800}

    with patch("app.services.vector_store.factory.AsyncPgVector") as MockPG:
        mock_embeddings = MagicMock()
        factory.get_vector_store(
            "conn", mock_embeddings, "coll", mode="async", pool_recycle=-1
        )
        _, kwargs = MockPG.call_args
        assert "pool_recycle" not in kwargs.get("engine_args", {})


def test_get_vector_store_schema_sets_search_path():
    """schema=<name> must translate into a connect_args search_path option so
    pgvector's tables are created and queried in the requested schema.
    `public` must be appended so the `vector` data type — installed by the
    extension into whichever schema it was created in (usually `public`) —
    remains resolvable; without it CREATE TABLE fails with
    `type "vector" does not exist`."""
    with patch(
        "app.services.vector_store.factory._verify_schemas_exist"
    ), patch("app.services.vector_store.factory.AsyncPgVector") as MockPG:
        mock_embeddings = MagicMock()
        factory.get_vector_store(
            "conn", mock_embeddings, "coll", mode="async", schema="myapp"
        )
        _, kwargs = MockPG.call_args
        engine_args = kwargs.get("engine_args")
        assert engine_args is not None
        assert engine_args["connect_args"]["options"] == "-csearch_path=myapp,public"
        # schema must not displace the pool defaults already on engine_args
        assert engine_args.get("pool_pre_ping") is True


def test_get_vector_store_schema_preserves_user_supplied_public():
    """If the user explicitly lists `public` we must not duplicate it."""
    with patch(
        "app.services.vector_store.factory._verify_schemas_exist"
    ), patch("app.services.vector_store.factory.AsyncPgVector") as MockPG:
        mock_embeddings = MagicMock()
        factory.get_vector_store(
            "conn", mock_embeddings, "coll", mode="async", schema="myapp,public"
        )
        _, kwargs = MockPG.call_args
        assert (
            kwargs["engine_args"]["connect_args"]["options"]
            == "-csearch_path=myapp,public"
        )


def test_get_vector_store_schema_accepts_multi_schema_list():
    """Comma-separated list for callers whose `vector` extension lives in a
    non-public schema (e.g. a dedicated `extensions` schema)."""
    with patch(
        "app.services.vector_store.factory._verify_schemas_exist"
    ), patch("app.services.vector_store.factory.AsyncPgVector") as MockPG:
        mock_embeddings = MagicMock()
        factory.get_vector_store(
            "conn",
            mock_embeddings,
            "coll",
            mode="async",
            schema="myapp, extensions",
        )
        _, kwargs = MockPG.call_args
        # whitespace around comma is stripped; public still auto-appended
        assert (
            kwargs["engine_args"]["connect_args"]["options"]
            == "-csearch_path=myapp,extensions,public"
        )


def test_get_vector_store_schema_verifies_existence_before_engine_args():
    """The factory must call _verify_schemas_exist with the parsed schemas
    before constructing the vector store, so a typo in POSTGRES_SCHEMA fails
    fast instead of silently creating tables in `public`."""
    with patch(
        "app.services.vector_store.factory._verify_schemas_exist"
    ) as mock_verify, patch(
        "app.services.vector_store.factory.AsyncPgVector"
    ):
        mock_embeddings = MagicMock()
        factory.get_vector_store(
            "conn://test",
            mock_embeddings,
            "coll",
            mode="async",
            schema="myapp, extensions",
        )
        mock_verify.assert_called_once_with("conn://test", ["myapp", "extensions"])


def test_get_vector_store_schema_validation_error_propagates():
    """If any configured schema is missing, _verify_schemas_exist raises and
    we must surface that — not quietly skip validation and let tables land
    in `public`."""
    with patch(
        "app.services.vector_store.factory._verify_schemas_exist",
        side_effect=ValueError("schema 'typo' does not exist"),
    ), patch("app.services.vector_store.factory.AsyncPgVector") as MockPG:
        mock_embeddings = MagicMock()
        with pytest.raises(ValueError, match="does not exist"):
            factory.get_vector_store(
                "conn", mock_embeddings, "coll", mode="async", schema="typo"
            )
        MockPG.assert_not_called()


def test_load_file_content_cleans_up_on_lazy_load_failure():
    """cleanup_temp_encoding_file is called even when lazy_load() raises."""
    from app.routes.document_routes import load_file_content

    mock_loader = MagicMock()
    mock_loader._temp_filepath = "/tmp/fake.csv"
    mock_loader.lazy_load.side_effect = RuntimeError("disk error")

    with patch(
        "app.routes.document_routes.get_loader",
        return_value=(mock_loader, True, "csv"),
    ):
        with patch(
            "app.routes.document_routes.cleanup_temp_encoding_file"
        ) as mock_cleanup:
            with pytest.raises(RuntimeError, match="disk error"):
                asyncio.run(
                    load_file_content("f.csv", "text/csv", "/fake/path", executor=None)
                )
            mock_cleanup.assert_called_once_with(mock_loader)
