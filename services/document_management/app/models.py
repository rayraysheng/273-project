from pydantic import BaseModel
from typing import Optional

class Document(BaseModel):
    title: str
    content: str
    author: Optional[str] = None

class DocumentInDB(Document):
    id: str  # MongoDB ID
