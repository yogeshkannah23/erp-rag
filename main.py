"""
PDF Vector Service — FastAPI entry point.

All business logic lives in application/use_cases/.
Infrastructure adapters live in infrastructure/.
API concerns (routers, schemas, DI) live in interfaces/.

Environment variables required:
  OPENAI_API_KEY         — OpenAI key for LLM
  HUGGINGFACE_API_TOKEN  — HuggingFace token for embeddings
  MILVUS_HOST            — Milvus hostname (default: localhost)
  MILVUS_PORT            — Milvus gRPC port (default: 19530)
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from interfaces.api.dependencies import get_vector_store
from interfaces.api.routers import chat, chunks, health, vectors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF Vector Service", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

app.include_router(chunks.router)
app.include_router(vectors.router)
app.include_router(chat.router)
app.include_router(health.router)


@app.on_event("startup")
def startup_check():
    ok, msg = get_vector_store().health_check()
    if ok:
        logger.info(f"[Startup] {msg}")
    else:
        logger.error(f"[Startup] {msg}")
