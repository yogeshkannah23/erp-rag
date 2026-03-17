"""
Infrastructure: HuggingFace Inference API embeddings adapter.
Implements both EmbeddingPort (domain) and LangChain Embeddings (for Milvus integration).
"""
import logging
from typing import List

from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings

import config
from domain.ports.embedding_port import EmbeddingPort

logger = logging.getLogger(__name__)


class HuggingFaceEmbeddings(EmbeddingPort, Embeddings):
    """Singleton HuggingFace embeddings client."""

    _instance = None
    _client: InferenceClient = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _get_client(self) -> InferenceClient:
        if self._client is None:
            self._client = InferenceClient(token=config.HUGGINGFACE_API_TOKEN)
        return self._client

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        client = self._get_client()
        results = []
        for text in texts:
            response = client.feature_extraction(text, model=config.EMBEDDING_MODEL_NAME)
            results.append(response.tolist() if hasattr(response, "tolist") else response)
        return results

    def embed_query(self, text: str) -> List[float]:
        response = self._get_client().feature_extraction(text, model=config.EMBEDDING_MODEL_NAME)
        return response.tolist() if hasattr(response, "tolist") else response
