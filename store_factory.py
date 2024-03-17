from langchain_community.embeddings import OpenAIEmbeddings

from store import AsnyPgVector, ExtendedPgVector


def get_vector_store(
    connection_string: str,
    embeddings: OpenAIEmbeddings,
    collection_name: str,
    mode: str = "sync",
):
    if mode == "sync":
        return ExtendedPgVector(
            connection_string=connection_string,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
    elif mode == "async":
        return AsnyPgVector(
            connection_string=connection_string,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
    else:
        raise ValueError("Invalid mode specified. Choose 'sync' or 'async'.")
