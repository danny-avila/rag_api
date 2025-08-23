# app/config.py
import os
import json
import boto3
import logging
import urllib.parse
from enum import Enum
from datetime import datetime
from dotenv import find_dotenv, load_dotenv
from starlette.middleware.base import BaseHTTPMiddleware

from app.services.vector_store.factory import get_vector_store

load_dotenv(find_dotenv())


class VectorDBType(Enum):
    PGVECTOR = "pgvector"
    ATLAS_MONGO = "atlas-mongo"


class EmbeddingsProvider(Enum):
    OPENAI = "openai"
    AZURE = "azure"
    HUGGINGFACE = "huggingface"
    HUGGINGFACETEI = "huggingfacetei"
    OLLAMA = "ollama"
    BEDROCK = "bedrock"
    GOOGLE_VERTEXAI = "vertexai"
    NVIDIA = "nvidia"


def get_env_variable(
    var_name: str, default_value: str = None, required: bool = False
) -> str:
    value = os.getenv(var_name)
    if value is None:
        if default_value is None and required:
            raise ValueError(f"Environment variable '{var_name}' not found.")
        return default_value
    # Strip comments and whitespace from environment variables
    if isinstance(value, str) and '#' in value:
        value = value.split('#')[0].strip()
    return value


RAG_HOST = os.getenv("RAG_HOST", "0.0.0.0")
RAG_PORT = int(os.getenv("RAG_PORT", 8000))

RAG_UPLOAD_DIR = get_env_variable("RAG_UPLOAD_DIR", "./uploads/")
if not os.path.exists(RAG_UPLOAD_DIR):
    os.makedirs(RAG_UPLOAD_DIR, exist_ok=True)

VECTOR_DB_TYPE = VectorDBType(
    get_env_variable("VECTOR_DB_TYPE", VectorDBType.PGVECTOR.value)
)
POSTGRES_USE_UNIX_SOCKET = (
    get_env_variable("POSTGRES_USE_UNIX_SOCKET", "False").lower() == "true"
)
POSTGRES_DB = get_env_variable("POSTGRES_DB", "mydatabase")
POSTGRES_USER = get_env_variable("POSTGRES_USER", "myuser")
POSTGRES_PASSWORD = get_env_variable("POSTGRES_PASSWORD", "mypassword")
DB_HOST = get_env_variable("DB_HOST", "db")
DB_PORT = get_env_variable("DB_PORT", "5432")
COLLECTION_NAME = get_env_variable("COLLECTION_NAME", "testcollection")
ATLAS_MONGO_DB_URI = get_env_variable(
    "ATLAS_MONGO_DB_URI", "mongodb://127.0.0.1:27018/LibreChat"
)
ATLAS_SEARCH_INDEX = get_env_variable("ATLAS_SEARCH_INDEX", "vector_index")
MONGO_VECTOR_COLLECTION = get_env_variable(
    "MONGO_VECTOR_COLLECTION", None
)  # Deprecated, backwards compatability
CHUNK_SIZE = int(get_env_variable("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(get_env_variable("CHUNK_OVERLAP", "100"))

env_value = get_env_variable("PDF_EXTRACT_IMAGES", "False").lower()
PDF_EXTRACT_IMAGES = True if env_value == "true" else False

if POSTGRES_USE_UNIX_SOCKET:
    connection_suffix = f"{urllib.parse.quote_plus(POSTGRES_USER)}:{urllib.parse.quote_plus(POSTGRES_PASSWORD)}@/{urllib.parse.quote_plus(POSTGRES_DB)}?host={urllib.parse.quote_plus(DB_HOST)}"
else:
    connection_suffix = f"{urllib.parse.quote_plus(POSTGRES_USER)}:{urllib.parse.quote_plus(POSTGRES_PASSWORD)}@{DB_HOST}:{DB_PORT}/{urllib.parse.quote_plus(POSTGRES_DB)}"

CONNECTION_STRING = f"postgresql+psycopg2://{connection_suffix}"
DSN = f"postgresql://{connection_suffix}"

## Logging

HTTP_RES = "http_res"
HTTP_REQ = "http_req"

logger = logging.getLogger()

debug_mode = os.getenv("DEBUG_RAG_API", "False").lower() in (
    "true",
    "1",
    "yes",
    "y",
    "t",
)
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

## Credentials

OPENAI_API_KEY = get_env_variable("OPENAI_API_KEY", "")
RAG_OPENAI_API_KEY = get_env_variable("RAG_OPENAI_API_KEY", OPENAI_API_KEY)
RAG_OPENAI_BASEURL = get_env_variable("RAG_OPENAI_BASEURL", None)
RAG_OPENAI_PROXY = get_env_variable("RAG_OPENAI_PROXY", None)
AZURE_OPENAI_API_KEY = get_env_variable("AZURE_OPENAI_API_KEY", "")
RAG_AZURE_OPENAI_API_VERSION = get_env_variable("RAG_AZURE_OPENAI_API_VERSION", None)
RAG_AZURE_OPENAI_API_KEY = get_env_variable(
    "RAG_AZURE_OPENAI_API_KEY", AZURE_OPENAI_API_KEY
)
AZURE_OPENAI_ENDPOINT = get_env_variable("AZURE_OPENAI_ENDPOINT", "")
RAG_AZURE_OPENAI_ENDPOINT = get_env_variable(
    "RAG_AZURE_OPENAI_ENDPOINT", AZURE_OPENAI_ENDPOINT
).rstrip("/")
HF_TOKEN = get_env_variable("HF_TOKEN", "")
OLLAMA_BASE_URL = get_env_variable("OLLAMA_BASE_URL", "http://ollama:11434")
AWS_ACCESS_KEY_ID = get_env_variable("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = get_env_variable("AWS_SECRET_ACCESS_KEY", "")
AWS_SESSION_TOKEN = get_env_variable("AWS_SESSION_TOKEN", "")
GOOGLE_APPLICATION_CREDENTIALS = get_env_variable("GOOGLE_APPLICATION_CREDENTIALS", "")
env_value = get_env_variable("RAG_CHECK_EMBEDDING_CTX_LENGTH", "True").lower()
RAG_CHECK_EMBEDDING_CTX_LENGTH = True if env_value == "true" else False

## Embeddings


def init_embeddings(provider, model):
    if provider == EmbeddingsProvider.OPENAI:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=model,
            api_key=RAG_OPENAI_API_KEY,
            openai_api_base=RAG_OPENAI_BASEURL,
            openai_proxy=RAG_OPENAI_PROXY,
            chunk_size=EMBEDDINGS_CHUNK_SIZE,
            check_embedding_ctx_length=RAG_CHECK_EMBEDDING_CTX_LENGTH,
        )
    elif provider == EmbeddingsProvider.AZURE:
        from langchain_openai import AzureOpenAIEmbeddings

        return AzureOpenAIEmbeddings(
            azure_deployment=model,
            api_key=RAG_AZURE_OPENAI_API_KEY,
            azure_endpoint=RAG_AZURE_OPENAI_ENDPOINT,
            api_version=RAG_AZURE_OPENAI_API_VERSION,
            chunk_size=EMBEDDINGS_CHUNK_SIZE,
            check_embedding_ctx_length=RAG_CHECK_EMBEDDING_CTX_LENGTH,
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

        return OllamaEmbeddings(model=model, base_url=OLLAMA_BASE_URL)
    elif provider == EmbeddingsProvider.GOOGLE_VERTEXAI:
        from langchain_google_vertexai import VertexAIEmbeddings

        return VertexAIEmbeddings(model=model)
    elif provider == EmbeddingsProvider.BEDROCK:
        from app.services.embeddings.bedrock_rate_limited import RateLimitedBedrockEmbeddings

        session_kwargs = {
            "aws_access_key_id": AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
            "region_name": AWS_DEFAULT_REGION,
        }

        if AWS_SESSION_TOKEN:
            session_kwargs["aws_session_token"] = AWS_SESSION_TOKEN

        session = boto3.Session(**session_kwargs)
        
        # Get reactive rate limiting configuration from environment
        max_batch = int(get_env_variable("BEDROCK_MAX_BATCH_SIZE", "15"))
        max_retries = int(get_env_variable("BEDROCK_MAX_RETRIES", "5"))
        initial_delay = float(get_env_variable("BEDROCK_INITIAL_RETRY_DELAY", "0.1"))
        max_delay = float(get_env_variable("BEDROCK_MAX_RETRY_DELAY", "30.0"))
        backoff_factor = float(get_env_variable("BEDROCK_BACKOFF_FACTOR", "2.0"))
        recovery_factor = float(get_env_variable("BEDROCK_RECOVERY_FACTOR", "0.9"))
        
        # Get Titan V2 specific parameters
        dimensions = get_env_variable("BEDROCK_EMBEDDING_DIMENSIONS", None)
        if dimensions is not None:
            dimensions = int(dimensions)
        normalize = get_env_variable("BEDROCK_EMBEDDING_NORMALIZE", "true").lower() == "true"
        
        # Create client with connection pooling for maximum performance
        config = boto3.session.Config(
            max_pool_connections=50,  # Increased for better concurrency
            retries={'max_attempts': 0}  # We handle retries in our wrapper
        )
        
        return RateLimitedBedrockEmbeddings(
            client=session.client("bedrock-runtime", config=config),
            model_id=model,
            region_name=AWS_DEFAULT_REGION,
            max_batch_size=max_batch,
            max_retries=max_retries,
            initial_retry_delay=initial_delay,
            max_retry_delay=max_delay,
            backoff_factor=backoff_factor,
            recovery_factor=recovery_factor,
            dimensions=dimensions,
            normalize=normalize,
        )
    elif provider == EmbeddingsProvider.NVIDIA:
        from app.services.embeddings.nvidia_embeddings import NVIDIAEmbeddings
        
        return NVIDIAEmbeddings(
            base_url=RAG_OPENAI_BASEURL,
            model=model,
            api_key=RAG_OPENAI_API_KEY,
            max_batch_size=int(get_env_variable("NVIDIA_MAX_BATCH_SIZE", "20")),
            max_retries=int(get_env_variable("NVIDIA_MAX_RETRIES", "3")),
            timeout=float(get_env_variable("NVIDIA_TIMEOUT", "30.0")),
            input_type=get_env_variable("NVIDIA_INPUT_TYPE", "query"),
            encoding_format=get_env_variable("NVIDIA_ENCODING_FORMAT", "float"),
            truncate=get_env_variable("NVIDIA_TRUNCATE", "NONE"),
        )
    else:
        raise ValueError(f"Unsupported embeddings provider: {provider}")


EMBEDDINGS_PROVIDER = EmbeddingsProvider(
    get_env_variable("EMBEDDINGS_PROVIDER", EmbeddingsProvider.OPENAI.value).lower()
)

if EMBEDDINGS_PROVIDER == EmbeddingsProvider.OPENAI:
    EMBEDDINGS_MODEL = get_env_variable("EMBEDDINGS_MODEL", "text-embedding-3-small")
    # 1000 is the default chunk size for OpenAI, but this causes API rate limits to be hit
    EMBEDDINGS_CHUNK_SIZE = get_env_variable("EMBEDDINGS_CHUNK_SIZE", 200)
elif EMBEDDINGS_PROVIDER == EmbeddingsProvider.AZURE:
    EMBEDDINGS_MODEL = get_env_variable("EMBEDDINGS_MODEL", "text-embedding-3-small")
    # 2048 is the default (and maximum) chunk size for Azure, but this often causes unexpected 429 errors
    EMBEDDINGS_CHUNK_SIZE = get_env_variable("EMBEDDINGS_CHUNK_SIZE", 200)
elif EMBEDDINGS_PROVIDER == EmbeddingsProvider.HUGGINGFACE:
    EMBEDDINGS_MODEL = get_env_variable(
        "EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
elif EMBEDDINGS_PROVIDER == EmbeddingsProvider.HUGGINGFACETEI:
    EMBEDDINGS_MODEL = get_env_variable(
        "EMBEDDINGS_MODEL", "http://huggingfacetei:3000"
    )
elif EMBEDDINGS_PROVIDER == EmbeddingsProvider.GOOGLE_VERTEXAI:
    EMBEDDINGS_MODEL = get_env_variable("EMBEDDINGS_MODEL", "text-embedding-004")
elif EMBEDDINGS_PROVIDER == EmbeddingsProvider.OLLAMA:
    EMBEDDINGS_MODEL = get_env_variable("EMBEDDINGS_MODEL", "nomic-embed-text")
elif EMBEDDINGS_PROVIDER == EmbeddingsProvider.BEDROCK:
    EMBEDDINGS_MODEL = get_env_variable(
        "EMBEDDINGS_MODEL", "amazon.titan-embed-text-v1"
    )
elif EMBEDDINGS_PROVIDER == EmbeddingsProvider.NVIDIA:
    EMBEDDINGS_MODEL = get_env_variable(
        "EMBEDDINGS_MODEL", "nvidia/llama-3.2-nemoretriever-300m-embed-v1"
    )
else:
    raise ValueError(f"Unsupported embeddings provider: {EMBEDDINGS_PROVIDER}")

# Load AWS credentials ONLY if Bedrock is used as primary or backup
backup_provider_str = get_env_variable("EMBEDDINGS_PROVIDER_BACKUP", None)
bedrock_needed = (
    EMBEDDINGS_PROVIDER == EmbeddingsProvider.BEDROCK or 
    (backup_provider_str and backup_provider_str.lower() == "bedrock")
)

if bedrock_needed:
    AWS_DEFAULT_REGION = get_env_variable("AWS_DEFAULT_REGION", "us-east-1")
    AWS_ACCESS_KEY_ID = get_env_variable("AWS_ACCESS_KEY_ID", None)
    AWS_SECRET_ACCESS_KEY = get_env_variable("AWS_SECRET_ACCESS_KEY", None)  
    AWS_SESSION_TOKEN = get_env_variable("AWS_SESSION_TOKEN", None)
    logger.debug("AWS credentials loaded for Bedrock provider")
else:
    # Set to None when not needed
    AWS_DEFAULT_REGION = None
    AWS_ACCESS_KEY_ID = None  
    AWS_SECRET_ACCESS_KEY = None
    AWS_SESSION_TOKEN = None
    logger.debug("AWS credentials not required - no Bedrock provider configured")

# Initialize embeddings with backup support
def init_embeddings_with_backup():
    """Initialize embeddings with automatic backup failover."""
    # Use already loaded backup provider string
    backup_model = get_env_variable("EMBEDDINGS_MODEL_BACKUP", None)
    
    if backup_provider_str and backup_model:
        # Backup is configured, create backup embeddings with failover
        backup_provider = EmbeddingsProvider(backup_provider_str.lower())
        
        logger.info(f"Backup provider configured: {backup_provider.value} / {backup_model}")
        
        try:
            # Initialize primary provider
            primary_embeddings = init_embeddings(EMBEDDINGS_PROVIDER, EMBEDDINGS_MODEL)
            logger.info(f"✅ Primary provider initialized: {EMBEDDINGS_PROVIDER.value}")
            
            try:
                # Initialize backup provider
                backup_embeddings = init_embeddings(backup_provider, backup_model)
                logger.info(f"✅ Backup provider initialized: {backup_provider.value}")
                
                # Create backup wrapper
                from app.services.embeddings.backup_embeddings import BackupEmbeddingsProvider
                
                return BackupEmbeddingsProvider(
                    primary_provider=primary_embeddings,
                    backup_provider=backup_embeddings,
                    primary_name=f"{EMBEDDINGS_PROVIDER.value}:{EMBEDDINGS_MODEL}",
                    backup_name=f"{backup_provider.value}:{backup_model}"
                )
                
            except Exception as backup_error:
                logger.warning(f"⚠️ Backup provider failed to initialize: {str(backup_error)}")
                logger.info(f"Continuing with primary provider only: {EMBEDDINGS_PROVIDER.value}")
                return primary_embeddings
                
        except Exception as primary_error:
            logger.error(f"❌ Primary provider failed to initialize: {str(primary_error)}")
            
            # Try to initialize backup as primary
            try:
                backup_embeddings = init_embeddings(backup_provider, backup_model)
                logger.warning(f"🔄 Using backup provider as primary: {backup_provider.value}")
                return backup_embeddings
            except Exception as backup_error:
                logger.error(f"❌ Both providers failed to initialize!")
                raise RuntimeError(
                    f"Failed to initialize any embedding provider. "
                    f"Primary ({EMBEDDINGS_PROVIDER.value}): {str(primary_error)}, "
                    f"Backup ({backup_provider.value}): {str(backup_error)}"
                ) from primary_error
    else:
        # No backup configured, use single provider
        return init_embeddings(EMBEDDINGS_PROVIDER, EMBEDDINGS_MODEL)

try:
    embeddings = init_embeddings_with_backup()
    logger.info(f"Initialized embeddings of type: {type(embeddings)}")
except Exception as e:
    error_message = str(e)
    
    # Provide helpful configuration error messages
    if EMBEDDINGS_PROVIDER == EmbeddingsProvider.BEDROCK:
        if "model identifier is invalid" in error_message:
            logger.error(
                f"❌ BEDROCK CONFIGURATION ERROR ❌\n\n"
                f"The Bedrock model '{EMBEDDINGS_MODEL}' is not available in region '{AWS_DEFAULT_REGION}'.\n\n"
                f"💡 Quick Fix:\n"
                f"   Set EMBEDDINGS_MODEL=amazon.titan-embed-text-v1 in your .env file\n\n"
                f"🔍 Available models in most regions:\n"
                f"   • amazon.titan-embed-text-v1\n"
                f"   • cohere.embed-english-v3\n"
                f"   • cohere.embed-multilingual-v3\n\n"
                f"🌍 To check available models in {AWS_DEFAULT_REGION}:\n"
                f"   AWS Console → Bedrock → Foundation models → Embedding"
            )
        elif "AccessDeniedException" in error_message:
            logger.error(
                f"❌ BEDROCK ACCESS ERROR ❌\n\n"
                f"Your AWS account doesn't have access to Bedrock in '{AWS_DEFAULT_REGION}'.\n\n"
                f"💡 Solutions:\n"
                f"   1. AWS Console → Bedrock → Model access → Request model access\n"
                f"   2. Enable foundation models you want to use\n"
                f"   3. Verify IAM permissions include 'bedrock:InvokeModel'\n\n"
                f"⚠️  Note: Bedrock may not be available in all regions"
            )
        else:
            logger.error(f"❌ BEDROCK ERROR: {error_message}")
    else:
        logger.error(f"❌ EMBEDDINGS ERROR ({EMBEDDINGS_PROVIDER}): {error_message}")
    
    raise RuntimeError(f"Failed to initialize embeddings: {error_message}") from e

# Vector store
if VECTOR_DB_TYPE == VectorDBType.PGVECTOR:
    vector_store = get_vector_store(
        connection_string=CONNECTION_STRING,
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        mode="async",
    )
elif VECTOR_DB_TYPE == VectorDBType.ATLAS_MONGO:
    # Backward compatability check
    if MONGO_VECTOR_COLLECTION:
        logger.info(
            f"DEPRECATED: Please remove env var MONGO_VECTOR_COLLECTION and instead use COLLECTION_NAME and ATLAS_SEARCH_INDEX. You can set both as same, but not neccessary. See README for more information."
        )
        ATLAS_SEARCH_INDEX = MONGO_VECTOR_COLLECTION
        COLLECTION_NAME = MONGO_VECTOR_COLLECTION
    vector_store = get_vector_store(
        connection_string=ATLAS_MONGO_DB_URI,
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        mode="atlas-mongo",
        search_index=ATLAS_SEARCH_INDEX,
    )
else:
    raise ValueError(f"Unsupported vector store type: {VECTOR_DB_TYPE}")

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
    "yml",
    "yaml",
    "eml",
    "ex",
    "exs",
    "erl",
    "tsx",
    "jsx",
    "lhs",
]
