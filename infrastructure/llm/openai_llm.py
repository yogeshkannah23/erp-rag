"""
Infrastructure: OpenAI LLM adapter — implements LLMPort via LangChain.
"""
import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import config
from domain.ports.llm_port import LLMPort

logger = logging.getLogger(__name__)


class OpenAILLM(LLMPort):
    _instance = None
    _llm: ChatOpenAI = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _get_llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=config.LLM_MODEL,
                temperature=config.LLM_TEMPERATURE,
                api_key=config.OPENAI_API_KEY,
            )
        return self._llm

    def generate(self, context: str, question: str, prompt_template: str) -> str:
        chain = (
            ChatPromptTemplate.from_template(prompt_template)
            | self._get_llm()
            | StrOutputParser()
        )
        return chain.invoke({"context": context, "question": question})

    async def agenerate(self, prompt: str) -> str:
        result = await self._get_llm().ainvoke(prompt)
        return result.content.strip() if hasattr(result, "content") else str(result).strip()

    def classify(self, text: str, system_prompt: str, user_prompt_template: str, **kwargs) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_prompt_template),
        ])
        chain = prompt | self._get_llm() | StrOutputParser()
        return chain.invoke({"query": text, **kwargs})
