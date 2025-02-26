import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from enum import Enum
import boto3
from app.store.vector import get_vector_store

load_dotenv(find_dotenv())

def get_env_variable(var_name: str, default_value: str = None, required: bool = False) -> str:
    value = os.getenv(var_name)
    if value is None:
        if default_value is None and required:
            raise ValueError(f"Environment variable '{var_name}' not found.")
        return default_value
    return value

# --- General Settings ---
RAG_HOST = os.getenv("RAG_HOST", "0.0.0.0")
RAG_PORT = int(os.getenv("RAG_PORT", 8000))

RAG_UPLOAD_DIR = get_env_variable("RAG_UPLOAD_DIR", "./uploads/")
if not os.path.exists(RAG_UPLOAD_DIR):
    os.makedirs(RAG_UPLOAD_DIR, exist_ok=True)

class VectorDBType(Enum):
    PGVECTOR = "pgvector"
    ATLAS_MONGO = "atlas-mongo"
    DUMMY = "dummy"

VECTOR_DB_TYPE = VectorDBType(get_env_variable("VECTOR_DB_TYPE", "pgvector"))

POSTGRES_DB = get_env_variable("POSTGRES_DB", "mydatabase")
POSTGRES_USER = get_env_variable("POSTGRES_USER", "myuser")
POSTGRES_PASSWORD = get_env_variable("POSTGRES_PASSWORD", "mypassword")
DB_HOST = get_env_variable("DB_HOST", "db")
DB_PORT = get_env_variable("DB_PORT", "5432")
COLLECTION_NAME = get_env_variable("COLLECTION_NAME", "testcollection")
ATLAS_MONGO_DB_URI = get_env_variable("ATLAS_MONGO_DB_URI", "mongodb://127.0.0.1:27018/LibreChat")
ATLAS_SEARCH_INDEX = get_env_variable("ATLAS_SEARCH_INDEX", "vector_index")
CHUNK_SIZE = int(get_env_variable("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(get_env_variable("CHUNK_OVERLAP", "100"))
PDF_EXTRACT_IMAGES = get_env_variable("PDF_EXTRACT_IMAGES", "False").lower() == "true"

POSTGRES_CONN_STRING = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{DB_HOST}:{DB_PORT}/{POSTGRES_DB}"
DSN = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{DB_HOST}:{DB_PORT}/{POSTGRES_DB}"

# --- Logging ---
HTTP_RES = "http_res"
HTTP_REQ = "http_req"

logger = logging.getLogger()
debug_mode = os.getenv("DEBUG_RAG_API", "False").lower() in ("true", "1", "yes", "y", "t")
console_json = get_env_variable("CONSOLE_JSON", "False").lower() == "true"

if debug_mode:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

if console_json:
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            json_record = {
                "message": record.getMessage(),
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "filename": record.filename,
                "lineno": record.lineno,
                "funcName": record.funcName,
                "module": record.module,
                "threadName": record.threadName,
            }
            if HTTP_REQ in record.__dict__:
                json_record[HTTP_REQ] = record.__dict__[HTTP_REQ]
            if HTTP_RES in record.__dict__:
                json_record[HTTP_RES] = record.__dict__[HTTP_RES]
            if record.levelno == logging.ERROR and record.exc_info:
                json_record["exception"] = self.formatException(record.exc_info)
            return json.dumps(json_record)
    formatter = JsonFormatter()
else:
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logging.getLogger("uvicorn.access").disabled = True

# --- Embeddings Initialization ---
from app.models import EmbeddingsProvider

def init_embeddings(provider, model):
    if provider == EmbeddingsProvider.OPENAI:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=model,
            api_key=get_env_variable("RAG_OPENAI_API_KEY", get_env_variable("OPENAI_API_KEY", "")),
            openai_api_base=get_env_variable("RAG_OPENAI_BASEURL", None),
            openai_proxy=get_env_variable("RAG_OPENAI_PROXY", None),
        )
    elif provider == EmbeddingsProvider.AZURE:
        from langchain_openai import AzureOpenAIEmbeddings
        return AzureOpenAIEmbeddings(
            azure_deployment=model,
            api_key=get_env_variable("RAG_AZURE_OPENAI_API_KEY", get_env_variable("AZURE_OPENAI_API_KEY", "")),
            azure_endpoint=get_env_variable("RAG_AZURE_OPENAI_ENDPOINT", "").rstrip("/"),
            api_version=get_env_variable("RAG_AZURE_OPENAI_API_VERSION", None),
        )
    elif provider == EmbeddingsProvider.HUGGINGFACE:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=model, encode_kwargs={"normalize_embeddings": True}
        )
    elif provider == EmbeddingsProvider.HUGGINGFACETEI:
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        return HuggingFaceEndpointEmbeddings(model=model)
    elif provider == EmbeddingsProvider.OLLAMA:
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=model, base_url=get_env_variable("OLLAMA_BASE_URL", "http://ollama:11434"))
    elif provider == EmbeddingsProvider.BEDROCK:
        from langchain_aws import BedrockEmbeddings
        AWS_DEFAULT_REGION = get_env_variable("AWS_DEFAULT_REGION", "us-east-1")
        session = boto3.Session(
            aws_access_key_id=get_env_variable("AWS_ACCESS_KEY_ID", ""),
            aws_secret_access_key=get_env_variable("AWS_SECRET_ACCESS_KEY", ""),
            region_name=AWS_DEFAULT_REGION,
        )
        return BedrockEmbeddings(
            client=session.client("bedrock-runtime"),
            model_id=model,
            region_name=AWS_DEFAULT_REGION,
        )
    else:
        raise ValueError(f"Unsupported embeddings provider: {provider}")

EMBEDDINGS_PROVIDER = EmbeddingsProvider(get_env_variable("EMBEDDINGS_PROVIDER", "openai").lower())

if EMBEDDINGS_PROVIDER == EmbeddingsProvider.OPENAI:
    EMBEDDINGS_MODEL = get_env_variable("EMBEDDINGS_MODEL", "text-embedding-3-small")
elif EMBEDDINGS_PROVIDER == EmbeddingsProvider.AZURE:
    EMBEDDINGS_MODEL = get_env_variable("EMBEDDINGS_MODEL", "text-embedding-3-small")
elif EMBEDDINGS_PROVIDER == EmbeddingsProvider.HUGGINGFACE:
    EMBEDDINGS_MODEL = get_env_variable("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
elif EMBEDDINGS_PROVIDER == EmbeddingsProvider.HUGGINGFACETEI:
    EMBEDDINGS_MODEL = get_env_variable("EMBEDDINGS_MODEL", "http://huggingfacetei:3000")
elif EMBEDDINGS_PROVIDER == EmbeddingsProvider.OLLAMA:
    EMBEDDINGS_MODEL = get_env_variable("EMBEDDINGS_MODEL", "nomic-embed-text")
elif EMBEDDINGS_PROVIDER == EmbeddingsProvider.BEDROCK:
    EMBEDDINGS_MODEL = get_env_variable("EMBEDDINGS_MODEL", "amazon.titan-embed-text-v1")
else:
    raise ValueError(f"Unsupported embeddings provider: {EMBEDDINGS_PROVIDER}")

embeddings = init_embeddings(EMBEDDINGS_PROVIDER, EMBEDDINGS_MODEL)
logger.info(f"Initialized embeddings of type: {type(embeddings)}")

# --- Vector Store Initialization ---
# Use dummy mode if VECTOR_STORE_MODE is set to "dummy", otherwise proceed as usual.
VECTOR_STORE_MODE = get_env_variable("VECTOR_STORE_MODE", None)

if VECTOR_DB_TYPE == VectorDBType.DUMMY:
    vector_store = get_vector_store(
        connection_string="dummy_conn",
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        mode="dummy",
    )
elif VECTOR_DB_TYPE == VectorDBType.PGVECTOR:
    vector_store = get_vector_store(
        connection_string=POSTGRES_CONN_STRING,
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        mode="async",
    )
elif VECTOR_DB_TYPE == VectorDBType.ATLAS_MONGO:
    vector_store = get_vector_store(
        connection_string=ATLAS_MONGO_DB_URI,
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        mode="atlas-mongo",
        search_index=ATLAS_SEARCH_INDEX,
    )
else:
    raise ValueError(f"Unsupported vector store type: {VECTOR_DB_TYPE}")

# --- Known Source Extensions ---
known_source_ext = [
    "go", "py", "java", "sh", "bat", "ps1", "cmd", "js", "ts", "css", "cpp", "hpp",
    "h", "c", "cs", "sql", "log", "ini", "pl", "pm", "r", "dart", "dockerfile", "env",
    "php", "hs", "hsc", "lua", "nginxconf", "conf", "m", "mm", "plsql", "perl", "rb",
    "rs", "db2", "scala", "bash", "swift", "vue", "svelte",
]