"""
Port (interface) for PDF text extraction.
"""
from abc import ABC, abstractmethod


class PdfParserPort(ABC):
    @abstractmethod
    def extract_text(self, pdf_bytes: bytes) -> str:
        """Extract plain text from raw PDF bytes."""
