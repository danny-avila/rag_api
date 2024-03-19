# config.py

import os
import logging
from dotenv import find_dotenv, load_dotenv
from langchain_openai import OpenAIEmbeddings
from store_factory import get_vector_store

load_dotenv(find_dotenv())

def get_env_variable(var_name: str, default_value: str = None) -> str:
    value = os.getenv(var_name)
    if value is None:
        if default_value is None:
            raise ValueError(f"Environment variable '{var_name}' not found.")
        return default_value
    return value

POSTGRES_DB = get_env_variable("POSTGRES_DB")
POSTGRES_USER = get_env_variable("POSTGRES_USER")
POSTGRES_PASSWORD = get_env_variable("POSTGRES_PASSWORD")
DB_HOST = get_env_variable("DB_HOST")
DB_PORT = get_env_variable("DB_PORT")
COLLECTION_NAME = get_env_variable("COLLECTION_NAME", "testcollection")

CHUNK_SIZE = int(get_env_variable("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(get_env_variable("CHUNK_OVERLAP", "100"))
UPLOAD_DIR = get_env_variable("UPLOAD_DIR", "./uploads/")
env_value = get_env_variable("PDF_EXTRACT_IMAGES", "False").lower()
PDF_EXTRACT_IMAGES = True if env_value == "true" else False

CONNECTION_STRING = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{DB_HOST}:{DB_PORT}/{POSTGRES_DB}"

logger = logging.getLogger()

debug_mode = get_env_variable("DEBUG", "False").lower() == "true"
if debug_mode:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()  # or logging.FileHandler("app.log")
handler.setFormatter(formatter)
logger.addHandler(handler)

OPENAI_API_KEY = get_env_variable("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings()

pgvector_store = get_vector_store(
    connection_string=CONNECTION_STRING,
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    mode="async",
)
retriever = pgvector_store.as_retriever()

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
