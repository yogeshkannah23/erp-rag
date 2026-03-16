"""
LLM Service — OpenAI via LangChain (singleton).
"""
import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config as app_config

logger = logging.getLogger(__name__)


class LLMService:
    _instance = None
    _llm = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=app_config.LLM_MODEL,
                temperature=app_config.LLM_TEMPERATURE,
                api_key=app_config.OPENAI_API_KEY,
            )
        return self._llm

    def generate_answer(self, context: str, question: str, prompt_template: str) -> str:
        llm = self.get_llm()
        chain = ChatPromptTemplate.from_template(prompt_template) | llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question})

    def classify_text(self, text: str, system_prompt: str, user_prompt_template: str, **kwargs) -> str:
        llm = self.get_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_prompt_template),
        ])
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"query": text, **kwargs})
