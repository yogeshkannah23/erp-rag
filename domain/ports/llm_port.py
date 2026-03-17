"""
Port (interface) for LLM operations.
"""
from abc import ABC, abstractmethod


class LLMPort(ABC):
    @abstractmethod
    def generate(self, context: str, question: str, prompt_template: str) -> str:
        """Generate an answer given context and a question."""

    @abstractmethod
    async def agenerate(self, prompt: str) -> str:
        """Asynchronously generate text from a raw prompt."""

    @abstractmethod
    def classify(self, text: str, system_prompt: str, user_prompt_template: str, **kwargs) -> str:
        """Run a classification / extraction prompt."""
