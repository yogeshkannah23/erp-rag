"""
PDF Vector Service — FastAPI

Endpoints:
  POST /process-pdf  — ingest a PDF, embed chunks, store in Milvus
  POST /chat         — chat with the knowledge base (RAG)
  POST /ask          — alias for /chat, returns full data envelope
  GET  /health       — health check

Environment variables required:
  OPENAI_API_KEY         — OpenAI key for LLM
  HUGGINGFACE_API_TOKEN  — HuggingFace token for embeddings
  MILVUS_HOST            — Milvus hostname (default: localhost)
  MILVUS_PORT            — Milvus gRPC port (default: 19530)
"""
import base64
import logging
import traceback
from io import BytesIO
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.documents import Document
from fastapi.middleware.cors import CORSMiddleware

from config import get_collection_name, check_milvus_connection
from services.pdf_processor import PDFProcessor
from services.vector_store_service import VectorStoreService
from core.rag_engine import RAGEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF Vector Service", version="1.0.0")


@app.on_event("startup")
def startup_check():
    ok, msg = check_milvus_connection()
    if ok:
        logger.info(f"[Startup] {msg}")
    else:
        logger.error(f"[Startup] {msg}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


# ── Schemas ────────────────────────────────────────────────────────────────────

class ProcessPDFRequest(BaseModel):
    file_content: str           # base64-encoded PDF bytes
    filename: str
    title: str
    domains: List[str]          # e.g. ["AI", "Backend"]
    technologies: str           # e.g. "Python, FastAPI"


class ProcessPDFResponse(BaseModel):
    success: bool
    message: str = ""
    chunks: int = 0
    collections: List[str] = []
    error: str = ""


class ChatRequest(BaseModel):
    message: str
    available_domains: Optional[List[str]] = None   # passed from ERPNext DB
    k: int = 4


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/process-pdf", response_model=ProcessPDFResponse)
async def process_pdf(request: ProcessPDFRequest) -> ProcessPDFResponse:
    logger.info(f"[/process-pdf] title={request.title!r} domains={request.domains}")
    try:
        pdf_bytes = BytesIO(base64.b64decode(request.file_content))

        chunks = await PDFProcessor().process_pdf_with_summarization(
            filename=request.filename,
            title=request.title,
            domain=request.domains[0],
            technologies=request.technologies,
            pdf_bytes=pdf_bytes,
        )

        if not chunks:
            logger.warning("[/process-pdf] 0 chunks produced — nothing stored in Milvus")

        vector_store = VectorStoreService()
        collections_added: List[str] = []
        collections_failed: List[str] = []

        for domain in request.domains:
            collection_name = get_collection_name(domain)
            try:
                domain_chunks = [
                    Document(
                        page_content=c.page_content,
                        metadata={**c.metadata, "domain": domain},
                    )
                    for c in chunks
                ]
                vector_store.add_documents(domain_chunks, collection_name)
                collections_added.append(collection_name)
            except Exception as e:
                collections_failed.append(collection_name)
                logger.error(f"[/process-pdf] Failed → {collection_name}: {e}\n{traceback.format_exc()}")

        logger.info(
            f"[/process-pdf] Done — {len(chunks)} chunks, "
            f"stored={collections_added}, failed={collections_failed}"
        )
        return ProcessPDFResponse(
            success=True,
            message=f"Processed {len(chunks)} chunks into {len(collections_added)} collections",
            chunks=len(chunks),
            collections=collections_added,
        )

    except Exception as e:
        logger.error(f"[/process-pdf] Error: {e}\n{traceback.format_exc()}")
        return ProcessPDFResponse(success=False, error=str(e))


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        result = RAGEngine().ask(
            question=request.message,
            available_domains=request.available_domains or None,
            k=request.k,
        )
        return {
            "success": True,
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "extracted_tags": result.get("extracted_tags", {}),
        }
    except Exception as e:
        error_message = str(e)
        logger.error(f"[/chat] {type(e).__name__}: {error_message}\n{traceback.format_exc()}")

        if "RESOURCE_EXHAUSTED" in error_message or "429" in error_message:
            user_message = "High demand — please try again in a moment."
        elif "quota" in error_message.lower():
            user_message = "API quota limit reached. Please try again later."
        elif "timeout" in error_message.lower():
            user_message = "Request timed out. Please try again."
        elif "connection" in error_message.lower():
            user_message = "Connection error. Please check your network."
        else:
            user_message = "An error occurred. Please try again."

        return {"success": False, "error": user_message, "answer": user_message}


@app.post("/ask")
def ask(request: ChatRequest):
    try:
        result = RAGEngine().ask(
            question=request.message,
            available_domains=request.available_domains or None,
            k=request.k,
        )
        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"[/ask] Error: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": str(e)}


@app.get("/health")
def health():
    milvus_ok, milvus_msg = check_milvus_connection()
    return {
        "status": "ok" if milvus_ok else "degraded",
        "milvus": {"connected": milvus_ok, "detail": milvus_msg},
    }
