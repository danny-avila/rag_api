# config.py
import json
import logging
from datetime import datetime

from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
    HuggingFaceHubEmbeddings,
    OllamaEmbeddings,
)
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from starlette.middleware.base import BaseHTTPMiddleware

from rag_api.config.settings import (
    ATLAS_MONGO_DB_URI,
    COLLECTION_NAME,
    CONNECTION_STRING,
    EMBEDDINGS_MODEL,
    EMBEDDINGS_PROVIDER,
    GOOGLE_API_KEY,
    HTTP_REQ,
    HTTP_RES,
    MONGO_VECTOR_COLLECTION,
    OLLAMA_BASE_URL,
    RAG_AZURE_OPENAI_API_KEY,
    RAG_AZURE_OPENAI_API_VERSION,
    RAG_AZURE_OPENAI_ENDPOINT,
    RAG_OPENAI_API_KEY,
    RAG_OPENAI_BASEURL,
    RAG_OPENAI_PROXY,
    SHUTTLEAI_KEY,
    VECTOR_DB_TYPE,
    EmbeddingsProvider,
    VectorDBType,
    get_env_variable,
)
from rag_api.db.store_factory import get_vector_store

logger = logging.getLogger()

debug_mode = get_env_variable("DEBUG_RAG_API", "False").lower() == "true"
console_json = get_env_variable("CONSOLE_JSON", "False").lower() == "true"

if debug_mode:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

if console_json:

    class JsonFormatter(logging.Formatter):
        def __init__(self):
            super(JsonFormatter, self).__init__()

        def format(self, record):
            json_record = {}

            json_record["message"] = record.getMessage()

            if HTTP_REQ in record.__dict__:
                json_record[HTTP_REQ] = record.__dict__[HTTP_REQ]

            if HTTP_RES in record.__dict__:
                json_record[HTTP_RES] = record.__dict__[HTTP_RES]

            if record.levelno == logging.ERROR and record.exc_info:
                json_record["exception"] = self.formatException(record.exc_info)

            timestamp = datetime.fromtimestamp(record.created)
            json_record["timestamp"] = timestamp.isoformat()

            # add level
            json_record["level"] = record.levelname
            json_record["filename"] = record.filename
            json_record["lineno"] = record.lineno
            json_record["funcName"] = record.funcName
            json_record["module"] = record.module
            json_record["threadName"] = record.threadName

            return json.dumps(json_record)

    formatter = JsonFormatter()
else:
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

handler = logging.StreamHandler()  # or logging.FileHandler("app.log")
handler.setFormatter(formatter)
logger.addHandler(handler)


class LogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)

        logger_method = logger.info

        if str(request.url).endswith("/health"):
            logger_method = logger.debug

        logger_method(
            f"Request {request.method} {request.url} - {response.status_code}",
            extra={
                HTTP_REQ: {"method": request.method, "url": str(request.url)},
                HTTP_RES: {"status_code": response.status_code},
            },
        )

        return response


logging.getLogger("uvicorn.access").disabled = True


def init_embeddings(provider, model):
    if provider == EmbeddingsProvider.OPENAI:
        return OpenAIEmbeddings(
            model=model,
            api_key=RAG_OPENAI_API_KEY,
            openai_api_base=RAG_OPENAI_BASEURL,
            openai_proxy=RAG_OPENAI_PROXY,
        )
    elif provider == EmbeddingsProvider.AZURE:
        return AzureOpenAIEmbeddings(
            azure_deployment=model,
            api_key=RAG_AZURE_OPENAI_API_KEY,
            azure_endpoint=RAG_AZURE_OPENAI_ENDPOINT,
            api_version=RAG_AZURE_OPENAI_API_VERSION,
        )
    elif provider == EmbeddingsProvider.HUGGINGFACE:
        return HuggingFaceEmbeddings(
            model_name=model, encode_kwargs={"normalize_embeddings": True}
        )
    elif provider == EmbeddingsProvider.HUGGINGFACETEI:
        return HuggingFaceHubEmbeddings(model=model)
    elif provider == EmbeddingsProvider.OLLAMA:
        return OllamaEmbeddings(model=model, base_url=OLLAMA_BASE_URL)
    elif provider == EmbeddingsProvider.GOOGLE:
        from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=model,
            api_key=GOOGLE_API_KEY,
        )
    elif provider == EmbeddingsProvider.VOYAGE:
        from langchain_voyageai import VoyageAIEmbeddings

        return VoyageAIEmbeddings(
            model=model,
        )
    elif provider == EmbeddingsProvider.SHUTTLEAI:
        return OpenAIEmbeddings(
            model=model,
            api_key=SHUTTLEAI_KEY,
            openai_api_base="https://api.shuttleai.app/v1",
        )
    elif provider == EmbeddingsProvider.COHERE:
        from langchain_cohere import CohereEmbeddings

        return CohereEmbeddings(
            model=model,
        )
    else:
        raise ValueError(f"Unsupported embeddings provider: {provider}")


embeddings = init_embeddings(EMBEDDINGS_PROVIDER, EMBEDDINGS_MODEL)

logger.info(f"Initialized embeddings of type: {type(embeddings)}")

# Vector store
if VECTOR_DB_TYPE == VectorDBType.PGVECTOR:
    vector_store = get_vector_store(
        connection_string=CONNECTION_STRING,
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        mode="async",
    )
elif VECTOR_DB_TYPE == VectorDBType.ATLAS_MONGO:
    logger.warning("Using Atlas MongoDB as vector store is not fully supported yet.")
    vector_store = get_vector_store(
        connection_string=ATLAS_MONGO_DB_URI,
        embeddings=embeddings,
        collection_name=MONGO_VECTOR_COLLECTION,
        mode="atlas-mongo",
    )
else:
    raise ValueError(f"Unsupported vector store type: {VECTOR_DB_TYPE}")

retriever = vector_store.as_retriever()
