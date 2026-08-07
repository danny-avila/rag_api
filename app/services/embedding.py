"""Query and candidate embedding against the vector store's own space.

Retrieval and ``fast-v1`` rerank share this cache on purpose: on the file-search
path the rerank query vector is the retrieval query vector, so the rerank costs
no inference at all beyond what retrieval already paid.
"""

import os
from functools import lru_cache
from typing import List

from app.config import EMBEDDINGS_MODEL, vector_store

QUERY_EMBEDDING_CACHE_SIZE = int(os.getenv("RAG_QUERY_EMBEDDING_CACHE_SIZE", "128"))

STORE_EMBEDDING_MODEL = EMBEDDINGS_MODEL


@lru_cache(maxsize=QUERY_EMBEDDING_CACHE_SIZE)
def get_cached_query_embedding(query: str) -> List[float]:
    return vector_store.embedding_function.embed_query(query)


def embed_texts(texts: List[str]) -> List[List[float]]:
    return vector_store.embedding_function.embed_documents(texts)
