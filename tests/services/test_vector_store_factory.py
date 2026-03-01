"""Tests for vector store factory shutdown and cleanup logic."""

from unittest.mock import MagicMock, patch

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
