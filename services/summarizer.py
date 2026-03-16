"""
Summarizer Service — uses LLM to condense 4 project sections into one summary.
"""
import logging
from typing import Dict

from .llm_service import LLMService
from config import SUMMARIZATION_PROMPT

logger = logging.getLogger(__name__)


class Summarizer:
    def __init__(self):
        self.llm_service = LLMService()

    async def summarize_project_sections(self, sections: Dict[str, str]) -> str:
        prompt_vars = {
            "business_problem": sections.get("business_problem", ""),
            "features": sections.get("features", ""),
            "tech_stack": sections.get("tech_stack", ""),
            "challenges": sections.get("key_challenges", ""),
        }
        llm = self.llm_service.get_llm()
        summary = await llm.ainvoke(SUMMARIZATION_PROMPT.format(**prompt_vars))
        return summary.content.strip() if hasattr(summary, "content") else str(summary).strip()
