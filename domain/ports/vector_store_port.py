"""
Port (interface) for vector store operations.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple

from domain.entities.chunk import Chunk


class VectorStorePort(ABC):
    @abstractmethod
    def add_documents(self, collection: str, chunks: List[Chunk]) -> List[str]:
        """Persist chunks into the given collection. Returns inserted IDs."""

    @abstractmethod
    def search(self, collection: str, query: str, k: int) -> List[Tuple[Chunk, float]]:
        """Similarity search. Returns (chunk, score) pairs sorted by relevance."""

    @abstractmethod
    def health_check(self) -> Tuple[bool, str]:
        """Returns (ok, message) after probing the store."""
