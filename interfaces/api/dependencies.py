"""
Dependency injection container.
All infrastructure singletons and use-case factories are wired here.
FastAPI endpoints use these functions with Depends().
"""
from functools import lru_cache

from application.use_cases.process_pdf_use_case import ProcessPdfUseCase
from application.use_cases.query_rag_use_case import QueryRagUseCase
from application.use_cases.upload_vectors_use_case import UploadVectorsUseCase
from infrastructure.classifier.llm_domain_classifier import LLMDomainClassifier
from infrastructure.embeddings.huggingface_embeddings import HuggingFaceEmbeddings
from infrastructure.llm.openai_llm import OpenAILLM
from infrastructure.pdf.pypdf_parser import PyPdfParser
from infrastructure.vector_store.milvus_vector_store import MilvusVectorStore


# ── Singletons ──────────────────────────────────────────────────────────────

@lru_cache
def get_llm() -> OpenAILLM:
    return OpenAILLM()


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings()


@lru_cache
def get_vector_store() -> MilvusVectorStore:
    return MilvusVectorStore()


@lru_cache
def get_pdf_parser() -> PyPdfParser:
    return PyPdfParser()


@lru_cache
def get_domain_classifier() -> LLMDomainClassifier:
    return LLMDomainClassifier(llm=get_llm())


# ── Use-case factories ───────────────────────────────────────────────────────

def get_process_pdf_use_case() -> ProcessPdfUseCase:
    return ProcessPdfUseCase(pdf_parser=get_pdf_parser(), llm=get_llm())


def get_upload_vectors_use_case() -> UploadVectorsUseCase:
    return UploadVectorsUseCase(vector_store=get_vector_store())


def get_query_rag_use_case() -> QueryRagUseCase:
    return QueryRagUseCase(
        vector_store=get_vector_store(),
        llm=get_llm(),
        classifier=get_domain_classifier(),
    )
