"""
Infrastructure: LLM-based domain classifier — implements DomainClassifierPort.
"""
import json
import logging
from typing import List, Optional

import config
from domain.ports.domain_classifier_port import DomainClassifierPort
from domain.ports.llm_port import LLMPort

logger = logging.getLogger(__name__)


class LLMDomainClassifier(DomainClassifierPort):
    def __init__(self, llm: LLMPort):
        self._llm = llm

    def classify(self, query: str, available_domains: Optional[List[str]] = None) -> dict:
        domains = available_domains or config.DEFAULT_DOMAINS
        domains_str = '", "'.join(domains)

        system_prompt = "Classify technical queries into domains. Return JSON only."
        user_prompt = (
            f'Available Domains: ["{domains_str}"]\n\n'
            "User Query: {query}\n\n"
            'Return: {{ "domains": ["Domain1"], "technologies": ["Tech1"] }}'
        )

        try:
            response = self._llm.classify(
                text=query,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt,
                query=query,
            )
            clean = response.strip().strip("```json").strip("```").strip()
            result = json.loads(clean)

            if "domains" in result:
                if not isinstance(result["domains"], list):
                    result["domains"] = [result["domains"]]
                result["domains"] = [d for d in result["domains"] if d in domains]

            logger.info(f"[LLMDomainClassifier] Domains: {result.get('domains')}")
            return result

        except Exception as e:
            logger.error(f"[LLMDomainClassifier] Classification failed: {e}")
            return {"domains": [], "technologies": []}
