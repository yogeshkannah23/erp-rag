"""
Pydantic request schemas for the API layer.
"""
from typing import List, Optional

from pydantic import BaseModel


class ChunkData(BaseModel):
    page_content: str
    metadata: dict


class CreateChunksRequest(BaseModel):
    file_content: str       # base64-encoded PDF bytes
    filename: str
    title: str
    domains: List[str]      # e.g. ["AI", "Backend"]
    technologies: str       # e.g. "Python, FastAPI"


class UploadVectorsRequest(BaseModel):
    chunks: List[ChunkData]
    domains: List[str] = []     # e.g. ["AI", "Backend"] — defaults to General if empty


class ChatRequest(BaseModel):
    message: str
    available_domains: Optional[List[str]] = None   # passed from ERPNext DB
    k: int = 4
