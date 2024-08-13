import os
from enum import Enum

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


def get_env_variable(
    var_name: str, default_value: str = None, required: bool = False
) -> str:
    """Retrieves an environment variable with optional default value and required flag."""
    value = os.getenv(var_name)
    if value is None:
        if default_value is None and required:
            raise ValueError(f"Environment variable '{var_name}' not found.")
        return default_value
    return value


class VectorDBType(Enum):
    PGVECTOR = "pgvector"
    ATLAS_MONGO = "atlas-mongo"


class EmbeddingsProvider(Enum):
    OPENAI = "openai"
    AZURE = "azure"
    HUGGINGFACE = "huggingface"
    HUGGINGFACETEI = "huggingfacetei"
    OLLAMA = "ollama"
    GOOGLE = "google"
    VOYAGE = "voyage"
    SHUTTLEAI = "shuttleai"
    COHERE = "cohere"


## Logging
HTTP_RES = "http_res"
HTTP_REQ = "http_req"

# RAG Server Configuration
RAG_HOST = os.getenv("RAG_HOST", "0.0.0.0")
RAG_PORT = int(os.getenv("RAG_PORT", 8000))

# Upload Directory
RAG_UPLOAD_DIR = get_env_variable("RAG_UPLOAD_DIR", "./uploads/")

# Vector Database Configuration
VECTOR_DB_TYPE = VectorDBType(
    get_env_variable("VECTOR_DB_TYPE", VectorDBType.PGVECTOR.value)
)

# Database Credentials (Adjust based on your database)
POSTGRES_DB = get_env_variable("POSTGRES_DB", "mydatabase")
POSTGRES_USER = get_env_variable("POSTGRES_USER", "myuser")
POSTGRES_PASSWORD = get_env_variable("POSTGRES_PASSWORD", "mypassword")
DB_HOST = get_env_variable("DB_HOST", "db")
DB_PORT = get_env_variable("DB_PORT", "5432")
COLLECTION_NAME = get_env_variable("COLLECTION_NAME", "testcollection")

# MongoDB Atlas Configuration (if using)
ATLAS_MONGO_DB_URI = get_env_variable(
    "ATLAS_MONGO_DB_URI", "mongodb://127.0.0.1:27018/LibreChat"
)
MONGO_VECTOR_COLLECTION = get_env_variable(
    "MONGO_VECTOR_COLLECTION", "vector_collection"
)

# Chunking Parameters
CHUNK_SIZE = int(get_env_variable("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(get_env_variable("CHUNK_OVERLAP", "100"))

# PDF Extraction
PDF_EXTRACT_IMAGES = get_env_variable("PDF_EXTRACT_IMAGES", "False").lower() == "true"

# Database Connection Strings
CONNECTION_STRING = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{DB_HOST}:{DB_PORT}/{POSTGRES_DB}"
DSN = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{DB_HOST}:{DB_PORT}/{POSTGRES_DB}"
print(CONNECTION_STRING)
# Credentials
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
GOOGLE_API_KEY = get_env_variable("GOOGLE_KEY", "")
VOYAGE_API_KEY = get_env_variable("VOYAGE_API_KEY", "")
SHUTTLEAI_KEY = get_env_variable("SHUTTLEAI_KEY", "")  # use embeddings from shuttleai
COHERE_API_KEY = get_env_variable("COHERE_API_KEY", "")

# Embeddings Configuration
EMBEDDINGS_PROVIDER = EmbeddingsProvider(
    get_env_variable("EMBEDDINGS_PROVIDER", EmbeddingsProvider.OPENAI.value).lower()
)

if EMBEDDINGS_PROVIDER == EmbeddingsProvider.OPENAI:
    EMBEDDINGS_MODEL = get_env_variable("EMBEDDINGS_MODEL", "text-embedding-3-small")
elif EMBEDDINGS_PROVIDER == EmbeddingsProvider.AZURE:
    EMBEDDINGS_MODEL = get_env_variable("EMBEDDINGS_MODEL", "text-embedding-3-small")
elif EMBEDDINGS_PROVIDER == EmbeddingsProvider.HUGGINGFACE:
    EMBEDDINGS_MODEL = get_env_variable(
        "EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
elif EMBEDDINGS_PROVIDER == EmbeddingsProvider.HUGGINGFACETEI:
    EMBEDDINGS_MODEL = get_env_variable(
        "EMBEDDINGS_MODEL", "http://huggingfacetei:3000"
    )
elif EMBEDDINGS_PROVIDER == EmbeddingsProvider.OLLAMA:
    EMBEDDINGS_MODEL = get_env_variable("EMBEDDINGS_MODEL", "nomic-embed-text")
elif EMBEDDINGS_PROVIDER == EmbeddingsProvider.GOOGLE:
    EMBEDDINGS_MODEL = get_env_variable("EMBEDDINGS_MODEL", "models/embedding-001")
elif EMBEDDINGS_PROVIDER == EmbeddingsProvider.VOYAGE:
    EMBEDDINGS_MODEL = get_env_variable("EMBEDDINGS_MODEL", "voyage-large-2")
elif EMBEDDINGS_PROVIDER == EmbeddingsProvider.SHUTTLEAI:
    # text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large
    EMBEDDINGS_MODEL = get_env_variable("EMBEDDINGS_MODEL", "text-embedding-3-large")
elif EMBEDDINGS_PROVIDER == EmbeddingsProvider.COHERE:
    EMBEDDINGS_MODEL = get_env_variable("EMBEDDINGS_MODEL", "embed-multilingual-v3.0")
else:
    raise ValueError(f"Unsupported embeddings provider: {EMBEDDINGS_PROVIDER}")

# Known Source File Extensions
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
