"""
Use Case: UploadVectorsUseCase
Embeds and stores pre-processed chunks into the vector store, per domain.
"""
import logging
import traceback
from typing import Dict, List

import config
from domain.entities.chunk import Chunk
from domain.ports.vector_store_port import VectorStorePort

logger = logging.getLogger(__name__)


class UploadVectorsUseCase:
    def __init__(self, vector_store: VectorStorePort):
        self._vector_store = vector_store

    def execute(self, chunks: List[Dict], domains: List[str]) -> Dict:
        """
        chunks: list of dicts with keys 'page_content' and 'metadata'
        domains: list of domain names to upload chunks into
        Returns: {'collections_added': [...], 'collections_failed': [...]}
        """
        if not domains:
            logger.warning("[UploadVectorsUseCase] No domains provided — falling back to 'General'")
            domains = ["General"]

        collections_added: List[str] = []
        collections_failed: List[str] = []

        for domain in domains:
            collection_name = config.get_collection_name(domain)
            try:
                domain_chunks = [
                    Chunk(
                        text=c["page_content"],
                        metadata={**c["metadata"], "domain": domain},
                    )
                    for c in chunks
                ]
                self._vector_store.add_documents(collection_name, domain_chunks)
                collections_added.append(collection_name)
                logger.info(f"[UploadVectorsUseCase] Stored {len(domain_chunks)} chunks → '{collection_name}'")
            except Exception as e:
                collections_failed.append(collection_name)
                logger.error(
                    f"[UploadVectorsUseCase] Failed → '{collection_name}': {e}\n"
                    f"{traceback.format_exc()}"
                )

        return {"collections_added": collections_added, "collections_failed": collections_failed}
