"""
Use Case: ProcessPdfUseCase
Orchestrates PDF text extraction → section parsing → LLM summarisation → chunking.
Depends only on domain ports — no infrastructure imports.
"""
import logging
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from domain.entities.chunk import Chunk
from domain.ports.llm_port import LLMPort
from domain.ports.pdf_parser_port import PdfParserPort
from domain.services.text_parser import TextParser

logger = logging.getLogger(__name__)


class ProcessPdfUseCase:
    def __init__(self, pdf_parser: PdfParserPort, llm: LLMPort):
        self._pdf_parser = pdf_parser
        self._llm = llm
        self._text_parser = TextParser()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )

    async def execute(
        self,
        pdf_bytes: bytes,
        filename: str,
        title: str,
        domain: str,
        technologies: str,
    ) -> List[Chunk]:
        text = self._pdf_parser.extract_text(pdf_bytes)
        if not text.strip():
            raise ValueError(f"No text extracted from PDF: {filename}")

        if not self._text_parser.has_required_sections(text):
            raise ValueError(
                f"PDF '{filename}' missing required sections "
                "(Business Problem / Features / Tech Stack / Challenges). "
                "Use the case-study template."
            )

        sections = self._text_parser.parse_project_document(text)
        logger.info(f"[ProcessPdfUseCase] Summarizing '{filename}'")

        summary = await self._llm.agenerate(
            config.SUMMARIZATION_PROMPT.format(
                business_problem=sections.get("business_problem", ""),
                features=sections.get("features", ""),
                tech_stack=sections.get("tech_stack", ""),
                challenges=sections.get("key_challenges", ""),
            )
        )

        metadata = {
            "source": filename,
            "title": title,
            "domain": domain,
            "technologies": technologies,
            "page": 1,
        }
        raw_chunks = self._splitter.split_text(summary)
        chunks = [Chunk(text=chunk, metadata=metadata) for chunk in raw_chunks]
        logger.info(f"[ProcessPdfUseCase] Done — {len(chunks)} chunks from '{filename}'")
        return chunks
