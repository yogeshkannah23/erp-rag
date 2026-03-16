"""
RAG Engine — retrieves relevant chunks from Milvus and generates an answer via LLM.
"""
import logging
from typing import List, Optional

from config import get_collection_name, RAG_PROMPT_TEMPLATE, CHUNKS_PER_PROJECT, MAX_TOTAL_CHUNKS
from services.vector_store_service import VectorStoreService
from services.llm_service import LLMService
from services.domain_classifier import DomainClassifier

logger = logging.getLogger(__name__)


class RAGEngine:
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.llm_service = LLMService()
        self.domain_classifier = DomainClassifier()

    def ask(self, question: str, available_domains: Optional[List[str]] = None, k: int = 20) -> dict:
        domain_info = self.domain_classifier.determine_domain(question, available_domains)
        domains = domain_info.get("domains", [])

        if not domains:
            return {
                "answer": (
                    "I couldn't identify a relevant domain for your query. "
                    "Please mention the technology area (e.g. Migration, AI, Security)."
                ),
                "extracted_tags": {},
                "sources": [],
            }

        all_results = []
        for domain in domains:
            collection_name = get_collection_name(domain)
            try:
                vectorstore = self.vector_store.get_vectorstore(collection_name)
                for doc, score in vectorstore.similarity_search_with_score(question, k=5):
                    all_results.append({
                        "doc": doc,
                        "score": score,
                        "project": doc.metadata.get("title", "Unknown"),
                        "domain": domain,
                    })
            except Exception as e:
                logger.error(f"[RAG] Search failed → {collection_name}: {e}")

        if not all_results:
            return {
                "answer": f"No case studies found for the domain(s): {', '.join(domains)}.",
                "extracted_tags": {
                    "searched_domains": domains,
                    "technologies": domain_info.get("technologies", []),
                },
                "sources": [],
            }

        all_results.sort(key=lambda x: x["score"])

        project_chunks: dict = {}
        for item in all_results:
            project_chunks.setdefault(item["project"], []).append(item)

        selected = []
        for chunks in project_chunks.values():
            selected.extend(chunks[:CHUNKS_PER_PROJECT])
        final = selected[:MAX_TOTAL_CHUNKS]

        context = "\n".join(
            f"Domain: {i['domain']} | Project: {i['project']}\n{i['doc'].page_content}\n---"
            for i in final
        )

        answer = self.llm_service.generate_answer(
            context=context,
            question=question,
            prompt_template=RAG_PROMPT_TEMPLATE,
        )

        return {
            "answer": answer,
            "extracted_tags": {
                "searched_domains": domains,
                "technologies": domain_info.get("technologies", []),
            },
            "sources": [
                {
                    "content": i["doc"].page_content[:150] + ("..." if len(i["doc"].page_content) > 150 else ""),
                    "project": i["project"],
                    "domain": i["domain"],
                }
                for i in final
            ],
        }
