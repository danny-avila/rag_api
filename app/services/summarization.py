import logging
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

MAP_PROMPT = ChatPromptTemplate.from_template(
    "Write a concise summary of the following text:\n\n{text}"
)

REDUCE_PROMPT = ChatPromptTemplate.from_template(
    "Combine these summaries into a consolidated summary:\n\n{summaries}"
)


def summarize_file_chunks(llm, documents: list[Document]) -> str:
    """Summarize a list of document chunks using map-reduce with parallel batch processing."""
    map_chain = MAP_PROMPT | llm | StrOutputParser()

    if len(documents) == 1:
        return map_chain.invoke({"text": documents[0].page_content})

    # Map: summarize all chunks in parallel using batch
    inputs = [{"text": doc.page_content} for doc in documents]
    chunk_summaries = map_chain.batch(inputs)

    # Reduce: combine all summaries
    reduce_chain = REDUCE_PROMPT | llm | StrOutputParser()
    combined = reduce_chain.invoke({"summaries": "\n\n".join(chunk_summaries)})
    return combined


def summarize_files(
    llm, grouped_documents: dict[str, list[Document]]
) -> list[dict]:
    """Summarize each file's chunks and return results."""
    results = []
    for file_id, docs in grouped_documents.items():
        logger.info(
            "Summarizing file %s (%d chunks)", file_id, len(docs)
        )
        summary = summarize_file_chunks(llm, docs)
        logger.info("Summary for file %s: %s", file_id, summary)
        results.append({
            "file_id": file_id,
            "summary": summary,
            "chunk_count": len(docs),
        })
    return results
