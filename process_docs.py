from langchain.schema import Document
import asyncio
from config import  (
    CONCURRENT_LIMIT,
    BATCH_SIZE,
    EMBEDDING_TIMEOUT,
    vector_store
)


#Prepare documents to be async added to vectorstore async in batches
async def store_documents(
    docs: list[Document], ids:list[str]
):
    semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)
    tasks = []
    #logger.info(f"Processing list of documents of length: {len(docs)}")
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i : min(i + BATCH_SIZE, len(docs))]
        #logger.info(f"Sending batch {i} to {i+len(batch)} / {len(docs)}")
        task = asyncio.create_task(process_batch(batch, ids, semaphore))
        tasks.append(task)
    try:
        idList = await asyncio.wait_for(asyncio.gather(*tasks), timeout=(EMBEDDING_TIMEOUT/1000)) 
    except asyncio.TimeoutError:
        raise Exception(f"TIMEOUT: embedding process took over the time limit of {EMBEDDING_TIMEOUT}ms. Partially added to database")
    return [id for sublist in idList for id in sublist]

#Helper for process_documents
async def process_batch(batch: list[Document], ids: list[str], semaphore):
    async with semaphore:
        return await vector_store.aadd_documents(batch,ids=ids)

