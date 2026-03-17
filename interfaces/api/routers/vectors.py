"""
Router: /upload-vectors
"""
import logging
import traceback

from fastapi import APIRouter, Depends

from application.use_cases.upload_vectors_use_case import UploadVectorsUseCase
from interfaces.api.dependencies import get_upload_vectors_use_case
from interfaces.schemas.requests import UploadVectorsRequest
from interfaces.schemas.responses import UploadVectorsResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload-vectors", response_model=UploadVectorsResponse)
def upload_vectors(
    request: UploadVectorsRequest,
    use_case: UploadVectorsUseCase = Depends(get_upload_vectors_use_case),
) -> UploadVectorsResponse:
    """Embed chunks returned by /create-chunks and store them in Milvus."""
    logger.info(f"[/upload-vectors] {len(request.chunks)} chunks, domains={request.domains}")
    try:
        chunks_data = [
            {"page_content": c.page_content, "metadata": c.metadata}
            for c in request.chunks
        ]
        result = use_case.execute(chunks=chunks_data, domains=request.domains)
        collections_added = result["collections_added"]
        logger.info(f"[/upload-vectors] Done — collections={collections_added}")
        return UploadVectorsResponse(
            success=True,
            message=f"Uploaded {len(request.chunks)} chunks into {len(collections_added)} collections",
            chunks=len(request.chunks),
            collections=collections_added,
        )
    except Exception as e:
        logger.error(f"[/upload-vectors] Error: {e}\n{traceback.format_exc()}")
        return UploadVectorsResponse(success=False, error=str(e))
