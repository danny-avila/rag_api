# app/constants.py
from enum import Enum

# Service endpoint limits — fixed by the search-stack contract, not tunable per
# deployment: callers batch against them and the projector sizes its work units
# from them.
MAX_EMBEDDING_INPUTS = 64
MAX_EMBEDDING_CHARS = 256_000
MAX_RERANK_CANDIDATES = 50
MAX_RERANK_TOP_N = 25

RERANK_PROFILE_FAST_V1 = "fast-v1"


class MESSAGES(str, Enum):
    DEFAULT = lambda msg="": f"{msg if msg else ''}"


class ERROR_MESSAGES(str, Enum):
    def __str__(self) -> str:
        return super().__str__()

    DEFAULT = lambda err="": f"Something went wrong :/\n{err if err else ''}"
    PANDOC_NOT_INSTALLED = "Pandoc is not installed on the server. Please contact your administrator for assistance."
    OPENAI_NOT_FOUND = lambda name="": f"OpenAI API was not found"
    OLLAMA_NOT_FOUND = "WebUI could not connect to Ollama"
    FILE_NOT_FOUND = "The specified file was not found."
