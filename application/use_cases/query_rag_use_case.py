"""
Use Case: QueryRagUseCase
Classifies the query → searches relevant collections → generates an LLM answer.
"""
import logging
from typing import List, Optional

import config
from domain.entities.query_result import QueryResult, Source
from domain.ports.domain_classifier_port import DomainClassifierPort
from domain.ports.llm_port import LLMPort
from domain.ports.vector_store_port import VectorStorePort

logger = logging.getLogger(__name__)


class QueryRagUseCase:
    def __init__(
        self,
        vector_store: VectorStorePort,
        llm: LLMPort,
        classifier: DomainClassifierPort,
    ):
        self._vector_store = vector_store
        self._llm = llm
        self._classifier = classifier

    def execute(
        self,
        question: str,
        available_domains: Optional[List[str]] = None,
        k: int = 5,
    ) -> QueryResult:
        domain_info = self._classifier.classify(question, available_domains)
        domains = domain_info.get("domains", [])

        if not domains:
            return QueryResult(
                answer=(
                    "I couldn't identify a relevant domain for your query. "
                    "Please mention the technology area (e.g. Migration, AI, Security)."
                ),
                sources=[],
                extracted_tags={},
            )

        all_results = []
        for domain in domains:
            collection_name = config.get_collection_name(domain)
            try:
                for chunk, score in self._vector_store.search(collection_name, question, k=5):
                    all_results.append({
                        "chunk": chunk,
                        "score": score,
                        "project": chunk.metadata.get("title", "Unknown"),
                        "domain": domain,
                    })
            except Exception as e:
                logger.error(f"[QueryRagUseCase] Search failed → '{collection_name}': {e}")

        if not all_results:
            return QueryResult(
                answer=f"No case studies found for the domain(s): {', '.join(domains)}.",
                sources=[],
                extracted_tags={
                    "searched_domains": domains,
                    "technologies": domain_info.get("technologies", []),
                },
            )

        all_results.sort(key=lambda x: x["score"])

        # Limit chunks per project to avoid a single project dominating the context
        project_chunks: dict = {}
        for item in all_results:
            project_chunks.setdefault(item["project"], []).append(item)

        selected = []
        for chunks in project_chunks.values():
            selected.extend(chunks[: config.CHUNKS_PER_PROJECT])
        final = selected[: config.MAX_TOTAL_CHUNKS]

        context = "\n".join(
            f"Domain: {i['domain']} | Project: {i['project']}\n{i['chunk'].text}\n---"
            for i in final
        )

        answer = self._llm.generate(
            context=context,
            question=question,
            prompt_template=config.RAG_PROMPT_TEMPLATE,
        )

        return QueryResult(
            answer=answer,
            sources=[
                Source(
                    content=i["chunk"].text[:150] + ("..." if len(i["chunk"].text) > 150 else ""),
                    project=i["project"],
                    domain=i["domain"],
                )
                for i in final
            ],
            extracted_tags={
                "searched_domains": domains,
                "technologies": domain_info.get("technologies", []),
            },
        )
