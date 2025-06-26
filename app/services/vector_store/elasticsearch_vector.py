from concurrent.futures import ThreadPoolExecutor
import threading
from typing import List, Tuple
from langchain_elasticsearch import ElasticsearchStore
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

class ExtendedElasticsearchVector(ElasticsearchStore):
    @property
    def embedding_function(self) -> Embeddings:
        """
        Property to access the embedding function.

        Returns:
            Embeddings: The embedding function used for generating vector representations.
        """
       
        return self.embedding

    def add_documents(self, docs: List[Document], ids: List[str]):
        """
        Adds a list of documents to the vector store after embedding them.

        Args:
            docs (List[Document]): A list of documents to be embedded and stored.
            ids (List[str]): Base identifier for the documents. Each doc will be assigned a unique ID.

        Returns:
            Any: Result of adding texts to the underlying store.
        """
       
        embedded_vectors = [None] * len(docs)
        lock = threading.Lock()

        # Worker function to embed documents in parallel
        def worker(i):
            text = docs[i].page_content
            emb = self.embedding.embed_query(text)
            with lock:
                embedded_vectors[i] = emb

        # Use a ThreadPoolExecutor to parallelize embedding computation
        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(worker, range(len(docs)))

        # Generate unique file IDs using the base ID
        file_ids = [f"{ids[0]}_{i}" for i in range(len(embedded_vectors))]

        return self._store.add_texts(
            ids=file_ids,
            texts=[doc.page_content for doc in docs],
            metadatas=[doc.metadata for doc in docs],
            vectors=embedded_vectors,
            create_index_if_not_exists=True,
            refresh_indices=False,
            bulk_kwargs={"chunk_size": 1000}
        )

    def similarity_search_with_score_by_vector(
        self, 
        query: str, 
        embedding: List[float], 
        file_ids: List[str], 
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """
        Performs a similarity search using a given embedding and returns results with scores.

        Args:
            query (str): Text query to include in the search (used in match clause).
            embedding (List[float]): Embedding vector to search against.
            file_ids (List[str]): List of file IDs to restrict the search.

        Returns:
            List[Tuple[Document, float]]: A list of (Document, score) tuples matching the query.
        """
        
        query_body = {
            "query": {
                "min_score": 1.5,
                "query": {
                    "bool": {
                        "should": [
                            {
                                "script_score": {
                                    "query": {"match_all": {}},
                                    "script": {
                                        "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                                        "params": {"query_vector": embedding}
                                    }
                                }
                            },
                            {
                                "match": {
                                    "text": {"query": query}
                                }
                            }
                        ],
                        "must": [
                            {
                                "terms": {
                                    "metadata.file_id.keyword": file_ids
                                }
                            }
                        ]
                    }
                }
            }
        }

        result = self.client.search(index=self._store.index, body=query_body)
        documents = []

        for hit in result["hits"]["hits"]:
            documents.append(
                (
                    Document(
                        page_content=hit["_source"]["text"],
                        metadata=hit["_source"]["metadata"]
                    ),
                    hit["_score"]
                )
            )

        return documents

    def get_all_ids(self) -> List[str]:
        """
        Retrieves all document IDs from the vector store.

        Returns:
            List[str]: A list of document IDs.
        """
        
        query = {
            "query": {
                "match_all": {}
            },
            "_source": False,
            "stored_fields": []
        }

        ids: List[str] = []

        result = self.client.search(index=self._store.index, body=query)
        for hit in result["hits"]["hits"]:
            ids.append(hit["_id"])

        return ids

    def get_documents_by_ids(self, ids: List[str]) -> List[Document]:
        """
        Retrieves documents from the vector store by their unique IDs.

        Args:
            ids (List[str]): List of document IDs to retrieve.

        Returns:
            List[Document]: A list of Document objects corresponding to the provided IDs.
        """
        
        if not ids:
            return []
        
        query = {
            "query": {
                "terms": {
                    "_id": ids
                }
            }
        }

        documents: List[Document] = []
        result = self.client.search(index=self._store.index, body=query)

        for hit in result["hits"]["hits"]:
            source = hit["_source"]
            doc = Document(
                page_content=source.get("text", ""),
                metadata=source.get("metadata", {})
            )
            documents.append(doc)

        return documents
    
    def get_filtered_ids(self, ids: List[str]) -> List[str]:
        """
        Returns file IDs filtered by the provided list of file IDs

        Args:
            ids (List[str]): List of file IDs to filter by.

        Returns:
            List[str]: A list of file IDs that exist in the document.
        """
        
        query = {
            "query": {
                "terms": {
                    "metadata.file_id.keyword": ids
                }
            },
            "size": 0,
            "aggs": {
                "unique_file_ids": {
                    "terms": {
                        "field": "metadata.file_id.keyword",
                        "size": len(ids)
                    }
                }
            }
        }

        result = self.client.search(index=self._store.index, body=query)
        filtered_ids = []

        if "aggregations" in result and "unique_file_ids" in result["aggregations"]:
            for bucket in result["aggregations"]["unique_file_ids"]["buckets"]:
                filtered_ids.append(bucket["key"])

        return filtered_ids
