# config.py

import os
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

CHUNK_SIZE = int(get_env_variable("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(get_env_variable("CHUNK_OVERLAP", "100"))

CONNECTION_STRING = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{DB_HOST}:{DB_PORT}/{POSTGRES_DB}"

OPENAI_API_KEY = get_env_variable("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings()

pgvector_store = get_vector_store(
    connection_string=CONNECTION_STRING,
    embeddings=embeddings,
    collection_name="testcollection",
    mode="async",
)
retriever = pgvector_store.as_retriever()
