import hashlib
from enum import Enum
from typing import Optional
from pydantic import BaseModel

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
    file_id: str
    query: str
    k: int = 4

class CleanupMethod(str, Enum):
    incremental = "incremental"
    full = "full"