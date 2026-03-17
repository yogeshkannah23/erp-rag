"""
Infrastructure: Milvus vector store adapter — implements VectorStorePort via LangChain.
"""
import logging
import traceback
from typing import List, Tuple

from langchain_core.documents import Document as LCDocument
from langchain_milvus import Milvus

import config
from domain.entities.chunk import Chunk
from domain.ports.vector_store_port import VectorStorePort
from infrastructure.embeddings.huggingface_embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class MilvusVectorStore(VectorStorePort):
    def __init__(self):
        logger.info("[MilvusVectorStore] Initializing — loading embedding model")
        self._embeddings = HuggingFaceEmbeddings()
        logger.info("[MilvusVectorStore] Ready")

    def _get_store(self, collection: str) -> Milvus:
        """Always creates a fresh Milvus instance to avoid stale connection issues."""
        from pymilvus import MilvusClient as PyMilvusClient, connections
        conn_args = config.get_milvus_connection_args()
        logger.info(f"[MilvusVectorStore] Connecting to {conn_args} for collection '{collection}'")
        try:
            _pre_client = PyMilvusClient(**conn_args)
            _alias = _pre_client._using
            if not connections.has_connection(_alias):
                connections.connect(alias=_alias, **conn_args)

            store = Milvus(
                embedding_function=self._embeddings,
                collection_name=collection,
                connection_args=conn_args,
                auto_id=True,
            )
            logger.info(f"[MilvusVectorStore] Store ready for '{collection}'")
            return store
        except Exception as e:
            logger.error(
                f"[MilvusVectorStore] Failed to create store for '{collection}': {e}\n"
                f"{traceback.format_exc()}"
            )
            raise

    def add_documents(self, collection: str, chunks: List[Chunk]) -> List[str]:
        if not chunks:
            logger.warning("[MilvusVectorStore] add_documents called with empty list — skipping")
            return []
        docs = [LCDocument(page_content=c.text, metadata=c.metadata) for c in chunks]
        store = self._get_store(collection)
        try:
            ids = store.add_documents(docs)
            logger.info(f"[MilvusVectorStore] Stored {len(ids)} vectors → '{collection}'")
            return ids
        except Exception as e:
            logger.error(
                f"[MilvusVectorStore] add_documents failed → '{collection}': {e}\n"
                f"{traceback.format_exc()}"
            )
            raise

    def search(self, collection: str, query: str, k: int) -> List[Tuple[Chunk, float]]:
        store = self._get_store(collection)
        results = store.similarity_search_with_score(query, k=k)
        return [
            (Chunk(text=doc.page_content, metadata=doc.metadata), score)
            for doc, score in results
        ]

    def health_check(self) -> Tuple[bool, str]:
        return config.check_milvus_connection()
