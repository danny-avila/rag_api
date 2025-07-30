# app/models.py
import hashlib
from enum import Enum
from pydantic import BaseModel
from typing import Optional, List


class DocumentResponse(BaseModel):
    page_content: str
    metadata: dict


class DocumentModel(BaseModel):
    page_content: str
    metadata: Optional[dict] = {}

    def generate_digest(self):
        hash_obj = hashlib.md5(self.page_content.encode())
        return hash_obj.hexdigest()


class StoreDocument(BaseModel):
    filepath: str
    filename: str
    file_content_type: str
    file_id: str


class QueryRequestBody(BaseModel):
    query: str
    file_id: Optional[str] = None
    k: int = 4
    entity_id: Optional[str] = None


class CleanupMethod(str, Enum):
    incremental = "incremental"
    full = "full"


class QueryMultipleBody(BaseModel):
    query: str
    file_ids: List[str]
    k: int = 4


# New KB-related models
class KBCreateResponse(BaseModel):
    kb_id: str
    collection_name: str
    status: str


class KBDeleteResponse(BaseModel):
    kb_id: str
    status: str


class KBInfoResponse(BaseModel):
    kb_id: str
    collection_id: str
    document_count: int


class V2EmbedResponse(BaseModel):
    kb_id: str
    file_id: str
    filename: str
    known_type: Optional[str]
    file_size_bytes: int
    chunk_count: int


class V2QueryResponse(BaseModel):
    results: List[dict]
    kb_id: str


class CrossKBQueryRequest(BaseModel):
    kb_ids: List[str]
    query: str
    k: int = 5


class CrossKBQueryResponse(BaseModel):
    results: List[dict]
    queried_kbs: List[str]
    total_results: int
