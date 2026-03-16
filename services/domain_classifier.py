"""
Domain Classifier Service — classifies a query into one or more domains using LLM.
Available domains are passed in from the caller (no database dependency).
"""
import json
import logging
from typing import List, Optional

from .llm_service import LLMService
from config import DEFAULT_DOMAINS

logger = logging.getLogger(__name__)


class DomainClassifier:
    def __init__(self):
        self.llm_service = LLMService()

    def determine_domain(self, query: str, available_domains: Optional[List[str]] = None) -> dict:
        domains = available_domains if available_domains else DEFAULT_DOMAINS
        domains_str = '", "'.join(domains)

        system_prompt = "Classify technical queries into domains. Return JSON only."
        user_prompt = (
            f'Available Domains: ["{domains_str}"]\n\n'
            "User Query: {query}\n\n"
            'Return: {{ "domains": ["Domain1"], "technologies": ["Tech1"] }}'
        )

        try:
            response = self.llm_service.classify_text(
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

            logger.info(f"[DomainClassifier] Domains: {result.get('domains')}")
            return result

        except Exception as e:
            logger.error(f"[DomainClassifier] Classification failed: {e}")
            return {"domains": [], "technologies": []}
