"""
Infrastructure: pypdf PDF text extractor — implements PdfParserPort.
"""
from io import BytesIO

from pypdf import PdfReader

from domain.ports.pdf_parser_port import PdfParserPort


class PyPdfParser(PdfParserPort):
    def extract_text(self, pdf_bytes: bytes) -> str:
        reader = PdfReader(BytesIO(pdf_bytes))
        return "".join(page.extract_text() or "" for page in reader.pages)
