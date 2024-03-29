# config.py
import json
import os
import logging
from datetime import datetime

from dotenv import find_dotenv, load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings, \
    HuggingFaceHubEmbeddings, OllamaEmbeddings
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from starlette.middleware.base import BaseHTTPMiddleware

from store_factory import get_vector_store

load_dotenv(find_dotenv())


def get_env_variable(var_name: str, default_value: str = None) -> str:
    value = os.getenv(var_name)
    if value is None:
        if default_value is None:
            raise ValueError(f"Environment variable '{var_name}' not found.")
        return default_value
    return value


RAG_UPLOAD_DIR = get_env_variable("RAG_UPLOAD_DIR", "./uploads/")
if not os.path.exists(RAG_UPLOAD_DIR):
    os.makedirs(RAG_UPLOAD_DIR, exist_ok=True)

POSTGRES_DB = get_env_variable("POSTGRES_DB", "mydatabase")
POSTGRES_USER = get_env_variable("POSTGRES_USER", "myuser")
POSTGRES_PASSWORD = get_env_variable("POSTGRES_PASSWORD", "mypassword")
DB_HOST = get_env_variable("DB_HOST", "db")
DB_PORT = get_env_variable("DB_PORT", "5432")
COLLECTION_NAME = get_env_variable("COLLECTION_NAME", "testcollection")

CHUNK_SIZE = int(get_env_variable("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(get_env_variable("CHUNK_OVERLAP", "100"))

env_value = get_env_variable("PDF_EXTRACT_IMAGES", "False").lower()
PDF_EXTRACT_IMAGES = True if env_value == "true" else False

CONNECTION_STRING = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{DB_HOST}:{DB_PORT}/{POSTGRES_DB}"
DSN = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{DB_HOST}:{DB_PORT}/{POSTGRES_DB}"

## Logging

HTTP_RES = "http_res"
HTTP_REQ = "http_req"

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
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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
                    HTTP_REQ: {"method": request.method,
                               "url": str(request.url)},
                    HTTP_RES: {"status_code": response.status_code},
                },
        )

        return response


logging.getLogger("uvicorn.access").disabled = True


## Credentials

OPENAI_API_KEY = get_env_variable("OPENAI_API_KEY", "")
AZURE_OPENAI_API_KEY = get_env_variable("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = get_env_variable("AZURE_OPENAI_ENDPOINT", "")
HF_TOKEN = get_env_variable("HF_TOKEN", "")
OLLAMA_BASE_URL = get_env_variable("OLLAMA_BASE_URL", "http://ollama:11434")


## Embeddings

def init_embeddings(provider, model):
    if provider == "openai":
        return OpenAIEmbeddings(model=model, api_key=OPENAI_API_KEY)
    elif provider == "azure":
        return AzureOpenAIEmbeddings(model=model,
                                     api_key=AZURE_OPENAI_API_KEY)  # AZURE_OPENAI_ENDPOINT is being grabbed from the environment
    elif provider == "huggingface":
        return HuggingFaceEmbeddings(model_name=model, encode_kwargs={
            'normalize_embeddings': True})
    elif provider == "huggingfacetei":
        return HuggingFaceHubEmbeddings(model=model)
    elif provider == "ollama":
        return OllamaEmbeddings(model=model, base_url=OLLAMA_BASE_URL)
    else:
        raise ValueError(f"Unsupported embeddings provider: {provider}")


EMBEDDINGS_PROVIDER = get_env_variable("EMBEDDINGS_PROVIDER", "openai").lower()

if EMBEDDINGS_PROVIDER == "openai":
    EMBEDDINGS_MODEL = get_env_variable("EMBEDDINGS_MODEL",
                                        "text-embedding-3-small")

elif EMBEDDINGS_PROVIDER == "azure":
    EMBEDDINGS_MODEL = get_env_variable("EMBEDDINGS_MODEL",
                                        "text-embedding-3-small")

elif EMBEDDINGS_PROVIDER == "huggingface":
    EMBEDDINGS_MODEL = get_env_variable("EMBEDDINGS_MODEL",
                                        "sentence-transformers/all-MiniLM-L6-v2")

elif EMBEDDINGS_PROVIDER == "huggingfacetei":
    EMBEDDINGS_MODEL = get_env_variable("EMBEDDINGS_MODEL",
                                        "http://huggingfacetei:3000")

elif EMBEDDINGS_PROVIDER == "ollama":
    EMBEDDINGS_MODEL = get_env_variable("EMBEDDINGS_MODEL", "nomic-embed-text")
else:
    raise ValueError(f"Unsupported embeddings provider: {EMBEDDINGS_PROVIDER}")

embeddings = init_embeddings(EMBEDDINGS_PROVIDER, EMBEDDINGS_MODEL)

logger.info(f"Initialized embeddings of type: {type(embeddings)}")

## Vector store

vector_store = get_vector_store(
        connection_string=CONNECTION_STRING,
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        mode="async",
)
retriever = vector_store.as_retriever()

known_source_ext = [
    "go",
    "py",
    "java",
    "sh",
    "bat",
    "ps1",
    "cmd",
    "js",
    "ts",
    "css",
    "cpp",
    "hpp",
    "h",
    "c",
    "cs",
    "sql",
    "log",
    "ini",
    "pl",
    "pm",
    "r",
    "dart",
    "dockerfile",
    "env",
    "php",
    "hs",
    "hsc",
    "lua",
    "nginxconf",
    "conf",
    "m",
    "mm",
    "plsql",
    "perl",
    "rb",
    "rs",
    "db2",
    "scala",
    "bash",
    "swift",
    "vue",
    "svelte",
]
