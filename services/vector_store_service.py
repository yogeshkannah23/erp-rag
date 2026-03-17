"""
Vector Store Service — Milvus via LangChain.
"""
import logging
import traceback
from typing import List, Optional

from langchain_core.documents import Document
from langchain_milvus import Milvus

from config import get_milvus_connection_args, get_collection_name
from .embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class VectorStoreService:
    def __init__(self):
        self.embeddings = EmbeddingService().get_embeddings()
        self._stores: dict = {}

    def get_vectorstore(self, collection_name: str) -> Milvus:
        if collection_name not in self._stores:
            store = Milvus(
                embedding_function=self.embeddings,
                collection_name=collection_name,
                connection_args=get_milvus_connection_args(),
                auto_id=True,
            )
            try:
                store._collection.load()
            except Exception:
                pass
            self._stores[collection_name] = store
            logger.info(f"[Milvus] Connected → {collection_name}")
        return self._stores[collection_name]

    def add_documents(self, documents: List[Document], collection_name: Optional[str] = None) -> List[str]:
        if not documents:
            return []
        if not collection_name:
            collection_name = get_collection_name(documents[0].metadata.get("domain", "general"))
        store = self.get_vectorstore(collection_name)
        try:
            ids = store.add_documents(documents)
            logger.info(f"[Milvus] Stored {len(ids)} vectors → {collection_name}")
            return ids
        except Exception as e:
            logger.error(f"[Milvus] Store failed → {collection_name}: {e}\n{traceback.format_exc()}")
            raise
