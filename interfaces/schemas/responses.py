"""
Pydantic response schemas for the API layer.
"""
from typing import List

from pydantic import BaseModel


class ChunkData(BaseModel):
    page_content: str
    metadata: dict


class CreateChunksResponse(BaseModel):
    success: bool
    message: str = ""
    chunk_count: int = 0
    chunks: List[ChunkData] = []
    error: str = ""


class UploadVectorsResponse(BaseModel):
    success: bool
    message: str = ""
    chunks: int = 0
    collections: List[str] = []
    error: str = ""
