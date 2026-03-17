"""
Vector Store Service — Milvus via LangChain.
"""
import logging
import traceback
from typing import List, Optional

from langchain_core.documents import Document
from langchain_milvus import Milvus
from pymilvus import connections as pymilvus_connections

from config import get_milvus_connection_args, get_collection_name
from .embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class VectorStoreService:
    def __init__(self):
        logger.info("[VectorStoreService] Initializing — loading embedding model")
        self.embeddings = EmbeddingService().get_embeddings()
        logger.info("[VectorStoreService] Embedding model loaded")
        self._stores: dict = {}

    def get_vectorstore(self, collection_name: str) -> Milvus:
        logger.info(f"[get_vectorstore] Requested collection: {collection_name}")

        if collection_name in self._stores:
            logger.info(f"[get_vectorstore] Returning cached store for: {collection_name}")
            return self._stores[collection_name]

        conn_args = get_milvus_connection_args()
        logger.info(f"[get_vectorstore] Connecting to Milvus at {conn_args}")

        try:
            pymilvus_connections.connect(alias="default", **conn_args)
            logger.info(f"[get_vectorstore] pymilvus connection established (alias=default)")
        except Exception as e:
            logger.error(f"[get_vectorstore] pymilvus connection failed: {e}\n{traceback.format_exc()}")
            raise

        logger.info(f"[get_vectorstore] Creating Milvus LangChain store for: {collection_name}")
        try:
            store = Milvus(
                embedding_function=self.embeddings,
                collection_name=collection_name,
                connection_args=conn_args,
                auto_id=True,
            )
            logger.info(f"[get_vectorstore] Milvus store created for: {collection_name}")
        except Exception as e:
            logger.error(f"[get_vectorstore] Failed to create Milvus store for {collection_name}: {e}\n{traceback.format_exc()}")
            raise

        self._stores[collection_name] = store
        logger.info(f"[get_vectorstore] Store cached — ready: {collection_name}")
        return store

    def add_documents(self, documents: List[Document], collection_name: Optional[str] = None) -> List[str]:
        logger.info(f"[add_documents] Called with {len(documents)} documents, collection={collection_name!r}")

        if not documents:
            logger.warning("[add_documents] No documents provided — skipping")
            return []

        if not collection_name:
            collection_name = get_collection_name(documents[0].metadata.get("domain", "general"))
            logger.info(f"[add_documents] Resolved collection name: {collection_name}")

        logger.info(f"[add_documents] Getting vectorstore for: {collection_name}")
        store = self.get_vectorstore(collection_name)

        logger.info(f"[add_documents] Calling store.add_documents() for: {collection_name}")
        try:
            ids = store.add_documents(documents)
            logger.info(f"[add_documents] Stored {len(ids)} vectors → {collection_name}")
            return ids
        except Exception as e:
            logger.error(f"[add_documents] store.add_documents() failed → {collection_name}: {e}\n{traceback.format_exc()}")
            raise
