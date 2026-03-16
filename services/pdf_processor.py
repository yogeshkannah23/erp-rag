"""
PDF Processing Service — extracts, parses, summarizes, and chunks a PDF.
"""
import logging
from io import BytesIO
from typing import List

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import CHUNK_SIZE, CHUNK_OVERLAP
from .text_parser import TextParser
from .summarizer import Summarizer

logger = logging.getLogger(__name__)


class PDFProcessor:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.text_parser = TextParser()
        self.summarizer = Summarizer()

    def _chunk(self, documents: List[Document]) -> List[Document]:
        return self.text_splitter.split_documents(documents)

    async def process_pdf_with_summarization(
        self,
        filename: str,
        title: str,
        domain: str,
        technologies: str,
        pdf_bytes: BytesIO,
    ) -> List[Document]:
        reader = PdfReader(pdf_bytes)
        full_text = "".join(p.extract_text() or "" for p in reader.pages)

        if not full_text.strip():
            raise ValueError(f"No text extracted from PDF: {filename}")

        if not self.text_parser.has_required_sections(full_text):
            raise ValueError(
                f"PDF '{filename}' missing required sections "
                "(Business Problem / Features / Tech Stack / Challenges). "
                "Use the case-study template."
            )

        sections = self.text_parser.parse_project_document(full_text)
        logger.info(f"[PDFProcessor] Summarizing {filename!r} — sections: {list(sections.keys())}")

        summary = await self.summarizer.summarize_project_sections(sections)
        chunks = self._chunk([Document(
            page_content=summary,
            metadata={
                "source": filename,
                "title": title,
                "domain": domain,
                "technologies": technologies,
                "page": 1,
            },
        )])
        logger.info(f"[PDFProcessor] Done — {len(chunks)} chunks from {filename!r}")
        return chunks
