"""
Port (interface) for embedding operations.
"""
from abc import ABC, abstractmethod
from typing import List


class EmbeddingPort(ABC):
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
