"""
Router: /create-chunks
"""
import base64
import logging
import traceback

from fastapi import APIRouter, Depends

from application.use_cases.process_pdf_use_case import ProcessPdfUseCase
from interfaces.api.dependencies import get_process_pdf_use_case
from interfaces.schemas.requests import CreateChunksRequest
from interfaces.schemas.responses import ChunkData, CreateChunksResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/create-chunks", response_model=CreateChunksResponse)
async def create_chunks(
    request: CreateChunksRequest,
    use_case: ProcessPdfUseCase = Depends(get_process_pdf_use_case),
) -> CreateChunksResponse:
    """Decode a base64 PDF, extract text, summarise, and return serialised chunks."""
    logger.info(f"[/create-chunks] title={request.title!r} domains={request.domains}")
    try:
        pdf_bytes = base64.b64decode(request.file_content)
        chunks = await use_case.execute(
            pdf_bytes=pdf_bytes,
            filename=request.filename,
            title=request.title,
            domain=request.domains[0],
            technologies=request.technologies,
        )
        serialised = [ChunkData(page_content=c.text, metadata=c.metadata) for c in chunks]
        logger.info(f"[/create-chunks] Done — {len(chunks)} chunks created")
        return CreateChunksResponse(
            success=True,
            message=f"Created {len(chunks)} chunks",
            chunk_count=len(chunks),
            chunks=serialised,
        )
    except Exception as e:
        logger.error(f"[/create-chunks] Error: {e}\n{traceback.format_exc()}")
        return CreateChunksResponse(success=False, error=str(e))
