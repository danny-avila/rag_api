import hashlib
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel

class DocumentResponse(BaseModel):
    page_content: str
    metadata: dict

class DocumentModel(BaseModel):
    page_content: str
    metadata: Optional[dict] = {}

    def generate_digest(self):
        return hashlib.md5(self.page_content.encode()).hexdigest()

class StoreDocument(BaseModel):
    filepath: str
    filename: str
    file_content_type: str
    file_id: str

class QueryRequestBody(BaseModel):
    query: str
    file_id: str
    k: int = 4
    entity_id: Optional[str] = None

class CleanupMethod(str, Enum):
    incremental = "incremental"
    full = "full"

class QueryMultipleBody(BaseModel):
    query: str
    file_ids: List[str]
    k: int = 4

class EmbeddingsProvider(str, Enum):
    OPENAI = "openai"
    AZURE = "azure"
    HUGGINGFACE = "huggingface"
    HUGGINGFACETEI = "huggingfacetei"
    OLLAMA = "ollama"
    BEDROCK = "bedrock"