# tests/conftest.py
import os

import pytest

# Set environment variables early so config picks up test settings.
os.environ["TESTING"] = "1"
# Set DB_HOST (and DSN) to dummy values to avoid real connection attempts.
os.environ["DB_HOST"] = "localhost"  # or any dummy value
os.environ["DSN"] = "dummy://"
# The application ships no fallback credentials, so the harness supplies its
# own throwaway values before app.config is imported. These reach a container
# that holds test data only.
os.environ.setdefault("POSTGRES_DB", "rag_api_test_db")
os.environ.setdefault("POSTGRES_USER", "rag_api_test_user")
os.environ.setdefault("POSTGRES_PASSWORD", "rag_api_test_password")

from app.services.vector_store.async_pg_vector import AsyncPgVector

# -- Patch the vector store classes to bypass DB connection --

# Do this *before* importing any app modules.
from langchain_community.vectorstores.pgvector import PGVector


def dummy_post_init(self):
    # Skip extension creation
    pass


# Integration tests that talk to a real container need the genuine bootstrap
# back, so keep a handle on it before it is replaced.
ORIGINAL_PGVECTOR_POST_INIT = PGVector.__post_init__

AsyncPgVector.__post_init__ = dummy_post_init
PGVector.__post_init__ = dummy_post_init

from langchain_core.documents import Document

from app import auth
from app.services import ratelimit


@pytest.fixture(autouse=True)
def reset_request_scoped_settings():
    """Auth and rate-limit settings are read once and cached, as in production.

    Tests mutate the environment, so the cache is dropped around every test.
    """
    auth.reset_settings()
    ratelimit.reset()
    yield
    auth.reset_settings()
    ratelimit.reset()


class DummyVectorStore:
    def get_all_ids(self, owners, tenants) -> list[str]:
        return ["testid1", "testid2"]

    def get_filtered_ids(self, ids, owners, tenants) -> list[str]:
        dummy_ids = ["testid1", "testid2"]
        return [id for id in dummy_ids if id in ids]

    async def get_documents_by_ids(self, ids, owners, tenants) -> list[Document]:
        return [
            Document(page_content="Test content", metadata={"file_id": id})
            for id in ids
        ]

    def delete_scoped(self, ids, owners, tenants) -> None:
        return None

    def similarity_search_with_score_by_vector(self, embedding, k: int, filter: dict):
        doc = Document(
            page_content="Queried content",
            metadata={
                "file_id": filter.get("file_id", "testid1"),
                "user_id": "testuser",
            },
        )
        return [(doc, 0.9)]

    def add_documents(self, documents, ids=None, **kwargs):
        return ids

    async def aadd_documents(self, documents, ids=None, **kwargs):
        return ids

    async def delete(self, ids=None, collection_only: bool = False):
        return None

    # Implement the missing as_retriever() method
    def as_retriever(self):
        # Return self or wrap with a dummy retriever if needed.
        return self
