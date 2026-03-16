"""
Embedding Service — HuggingFace Inference API (singleton).
"""
import logging
from typing import List

from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings

from config import HUGGINGFACE_API_TOKEN, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


class HuggingFaceAPIEmbeddings(Embeddings):
    def __init__(self, api_token: str, model_name: str):
        self.client = InferenceClient(token=api_token)
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            response = self.client.feature_extraction(text, model=self.model_name)
            embeddings.append(response.tolist() if hasattr(response, "tolist") else response)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        response = self.client.feature_extraction(text, model=self.model_name)
        return response.tolist() if hasattr(response, "tolist") else response


class EmbeddingService:
    _instance = None
    _embeddings = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_embeddings(self) -> HuggingFaceAPIEmbeddings:
        if self._embeddings is None:
            self._embeddings = HuggingFaceAPIEmbeddings(
                api_token=HUGGINGFACE_API_TOKEN,
                model_name=EMBEDDING_MODEL_NAME,
            )
        return self._embeddings
