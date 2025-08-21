# tests/dash_assistant/test_embeddings.py
"""Tests for embeddings functionality with deterministic mock embedder."""
import pytest
import numpy as np
from unittest.mock import patch
from app.dash_assistant.indexing.embedder import MockEmbedder, OpenAIEmbedder, get_embedder
from app.dash_assistant.indexing.index_jobs import IndexJob
from app.dash_assistant.db import DashAssistantDB


class TestMockEmbedder:
    """Test MockEmbedder for deterministic behavior."""
    
    def test_mock_embedder_deterministic(self):
        """Test that MockEmbedder produces deterministic vectors."""
        embedder = MockEmbedder(seed=42, dimension=3072)
        
        # Same content should produce same embedding
        content1 = "Revenue analytics dashboard"
        embedding1 = embedder.embed_text(content1)
        embedding2 = embedder.embed_text(content1)
        
        assert np.array_equal(embedding1, embedding2)
        assert len(embedding1) == 3072
        assert embedding1.dtype == np.float32
    
    def test_mock_embedder_different_content(self):
        """Test that different content produces different embeddings."""
        embedder = MockEmbedder(seed=42, dimension=3072)
        
        content1 = "Revenue analytics dashboard"
        content2 = "User retention metrics"
        
        embedding1 = embedder.embed_text(content1)
        embedding2 = embedder.embed_text(content2)
        
        assert not np.array_equal(embedding1, embedding2)
        assert len(embedding1) == len(embedding2) == 3072
    
    def test_mock_embedder_different_seeds(self):
        """Test that different seeds produce different embeddings for same content."""
        content = "Revenue analytics dashboard"
        
        embedder1 = MockEmbedder(seed=42, dimension=3072)
        embedder2 = MockEmbedder(seed=123, dimension=3072)
        
        embedding1 = embedder1.embed_text(content)
        embedding2 = embedder2.embed_text(content)
        
        assert not np.array_equal(embedding1, embedding2)
        assert len(embedding1) == len(embedding2) == 3072
    
    def test_mock_embedder_batch(self):
        """Test batch embedding functionality."""
        embedder = MockEmbedder(seed=42, dimension=3072)
        
        texts = [
            "Revenue analytics dashboard",
            "User retention metrics",
            "Product usage statistics"
        ]
        
        embeddings = embedder.embed_batch(texts)
        
        assert len(embeddings) == 3
        assert all(len(emb) == 3072 for emb in embeddings)
        assert all(emb.dtype == np.float32 for emb in embeddings)
        
        # Each embedding should be different
        assert not np.array_equal(embeddings[0], embeddings[1])
        assert not np.array_equal(embeddings[1], embeddings[2])
    
    def test_mock_embedder_vector_properties(self):
        """Test that mock embeddings have reasonable vector properties."""
        embedder = MockEmbedder(seed=42, dimension=3072)
        
        content = "Revenue analytics dashboard"
        embedding = embedder.embed_text(content)
        
        # Check vector is normalized (unit length)
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-5, f"Vector should be normalized, got norm: {norm}"
        
        # Check values are in reasonable range
        assert np.all(embedding >= -1.0) and np.all(embedding <= 1.0)


class TestEmbedderFactory:
    """Test embedder factory functionality."""
    
    @patch.dict('os.environ', {'EMBEDDINGS_PROVIDER': 'mock', 'EMBEDDINGS_DIM': '3072'})
    def test_get_mock_embedder(self):
        """Test getting mock embedder from factory."""
        embedder = get_embedder()
        assert isinstance(embedder, MockEmbedder)
        assert embedder.dimension == 3072
    
    @patch.dict('os.environ', {'EMBEDDINGS_PROVIDER': 'openai', 'EMBEDDINGS_DIM': '1536'})
    def test_get_openai_embedder(self):
        """Test getting OpenAI embedder from factory."""
        embedder = get_embedder()
        assert isinstance(embedder, OpenAIEmbedder)


@pytest.mark.asyncio
class TestIndexMissingChunks:
    """Test indexing missing chunks functionality."""
    
    async def test_index_missing_chunks_mock_provider(self, dash_assistant_db):
        """Test index_missing_chunks with mock embeddings provider."""
        # Setup test data - create chunks without embeddings
        await DashAssistantDB.execute_query("""
            INSERT INTO bi_entity (entity_type, title, description)
            VALUES ('dashboard', 'Test Dashboard', 'Test description')
        """)
        
        entity_id = await DashAssistantDB.fetch_value(
            "SELECT entity_id FROM bi_entity WHERE title = 'Test Dashboard'"
        )
        
        # Insert chunks without embeddings
        chunk_contents = [
            "Revenue analytics dashboard",
            "User retention metrics", 
            "Product usage statistics"
        ]
        
        for content in chunk_contents:
            await DashAssistantDB.execute_query("""
                INSERT INTO bi_chunk (entity_id, scope, content, lang)
                VALUES ($1, 'desc', $2, 'en')
            """, entity_id, content)
        
        # Verify chunks have no embeddings
        chunks_without_embeddings = await DashAssistantDB.fetch_all(
            "SELECT chunk_id FROM bi_chunk WHERE embedding IS NULL"
        )
        assert len(chunks_without_embeddings) == 3
        
        # Run index job with mock provider
        with patch.dict('os.environ', {'EMBEDDINGS_PROVIDER': 'mock', 'EMBEDDINGS_DIM': '3072'}):
            job = IndexJob()
            processed_count = await job.index_missing_chunks(batch_size=2)
        
        # Verify all chunks now have embeddings
        chunks_with_embeddings = await DashAssistantDB.fetch_all(
            "SELECT chunk_id, embedding FROM bi_chunk WHERE embedding IS NOT NULL"
        )
        assert len(chunks_with_embeddings) == 3
        assert processed_count == 3
        
        # Verify embeddings are 3072-dimensional
        for chunk in chunks_with_embeddings:
            embedding = chunk['embedding']
            assert len(embedding) == 3072
    
    async def test_index_missing_chunks_batch_processing(self, dash_assistant_db):
        """Test batch processing of missing chunks."""
        # Setup test data - create 5 chunks without embeddings
        await DashAssistantDB.execute_query("""
            INSERT INTO bi_entity (entity_type, title, description)
            VALUES ('dashboard', 'Batch Test Dashboard', 'Test description')
        """)
        
        entity_id = await DashAssistantDB.fetch_value(
            "SELECT entity_id FROM bi_entity WHERE title = 'Batch Test Dashboard'"
        )
        
        # Insert 5 chunks without embeddings
        for i in range(5):
            await DashAssistantDB.execute_query("""
                INSERT INTO bi_chunk (entity_id, scope, content, lang)
                VALUES ($1, 'desc', $2, 'en')
            """, entity_id, f"Test content {i}")
        
        # Run index job with batch_size=2
        with patch.dict('os.environ', {'EMBEDDINGS_PROVIDER': 'mock', 'EMBEDDINGS_DIM': '3072'}):
            job = IndexJob()
            processed_count = await job.index_missing_chunks(batch_size=2)
        
        # Verify all 5 chunks were processed
        assert processed_count == 5
        
        # Verify all chunks now have embeddings
        chunks_with_embeddings = await DashAssistantDB.fetch_all(
            "SELECT chunk_id FROM bi_chunk WHERE embedding IS NOT NULL"
        )
        assert len(chunks_with_embeddings) == 5
    
    async def test_index_missing_chunks_deterministic(self, dash_assistant_db):
        """Test that indexing produces deterministic results."""
        # Setup test data
        await DashAssistantDB.execute_query("""
            INSERT INTO bi_entity (entity_type, title, description)
            VALUES ('dashboard', 'Deterministic Test', 'Test description')
        """)
        
        entity_id = await DashAssistantDB.fetch_value(
            "SELECT entity_id FROM bi_entity WHERE title = 'Deterministic Test'"
        )
        
        # Insert chunk without embedding
        content = "Revenue analytics dashboard"
        await DashAssistantDB.execute_query("""
            INSERT INTO bi_chunk (entity_id, scope, content, lang)
            VALUES ($1, 'desc', $2, 'en')
        """, entity_id, content)
        
        # Run index job twice with same seed
        with patch.dict('os.environ', {'EMBEDDINGS_PROVIDER': 'mock', 'EMBEDDINGS_DIM': '3072', 'EMBEDDINGS_SEED': '42'}):
            job1 = IndexJob()
            await job1.index_missing_chunks(batch_size=10)
            
            # Get first embedding
            embedding1 = await DashAssistantDB.fetch_value(
                "SELECT embedding FROM bi_chunk WHERE content = $1", content
            )
            
            # Clear embedding and run again
            await DashAssistantDB.execute_query(
                "UPDATE bi_chunk SET embedding = NULL WHERE content = $1", content
            )
            
            job2 = IndexJob()
            await job2.index_missing_chunks(batch_size=10)
            
            # Get second embedding
            embedding2 = await DashAssistantDB.fetch_value(
                "SELECT embedding FROM bi_chunk WHERE content = $1", content
            )
        
        # Embeddings should be identical
        assert np.array_equal(np.array(embedding1), np.array(embedding2))
    
    async def test_index_missing_chunks_no_missing(self, dash_assistant_db):
        """Test index_missing_chunks when no chunks are missing embeddings."""
        with patch.dict('os.environ', {'EMBEDDINGS_PROVIDER': 'mock', 'EMBEDDINGS_DIM': '3072'}):
            job = IndexJob()
            processed_count = await job.index_missing_chunks(batch_size=10)
        
        # Should process 0 chunks when none are missing embeddings
        assert processed_count == 0
