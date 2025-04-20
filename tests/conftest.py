# tests/conftest.py
import os

from app.services.vector_store.async_pg_vector import AsyncPgVector

# Set environment variables early so config picks up test settings.
os.environ["TESTING"] = "1"
# Set DB_HOST (and DSN) to dummy values to avoid real connection attempts.
os.environ["DB_HOST"] = "localhost"  # or any dummy value
os.environ["DSN"] = "dummy://"

# -- Patch the vector store classes to bypass DB connection --

# Do this *before* importing any app modules.
from langchain_community.vectorstores.pgvector import PGVector

def dummy_post_init(self):
    # Skip extension creation
    pass

AsyncPgVector.__post_init__ = dummy_post_init
PGVector.__post_init__ = dummy_post_init

from langchain_core.documents import Document

class DummyVectorStore:
    def get_all_ids(self) -> list[str]:
        return ["testid1", "testid2"]
    
    def get_filtered_ids(self, ids) -> list[str]:
        dummy_ids = ["testid1", "testid2"]
        return [id for id in dummy_ids if id in ids]

    async def get_documents_by_ids(self, ids: list[str]) -> list[Document]:
        return [
            Document(page_content="Test content", metadata={"file_id": id})
            for id in ids
        ]

    def similarity_search_with_score_by_vector(self, embedding, k: int, filter: dict):
        doc = Document(
            page_content="Queried content",
            metadata={"file_id": filter.get("file_id", "testid1"), "user_id": "testuser"},
        )
        return [(doc, 0.9)]

    def add_documents(self, docs, ids):
        return ids

    async def aadd_documents(self, docs, ids):
        return ids

    async def delete(self, ids=None, collection_only: bool = False):
        return None

    # Implement the missing as_retriever() method
    def as_retriever(self):
        # Return self or wrap with a dummy retriever if needed.
        return self
