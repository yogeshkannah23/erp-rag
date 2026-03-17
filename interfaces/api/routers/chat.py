"""
Routers: /chat and /ask
"""
import logging
import traceback

from fastapi import APIRouter, Depends

from application.use_cases.query_rag_use_case import QueryRagUseCase
from interfaces.api.dependencies import get_query_rag_use_case
from interfaces.schemas.requests import ChatRequest

router = APIRouter()
logger = logging.getLogger(__name__)


def _result_to_dict(result) -> dict:
    return {
        "answer": result.answer,
        "sources": [
            {"content": s.content, "project": s.project, "domain": s.domain}
            for s in result.sources
        ],
        "extracted_tags": result.extracted_tags,
    }


@router.post("/chat")
def chat(
    request: ChatRequest,
    use_case: QueryRagUseCase = Depends(get_query_rag_use_case),
):
    try:
        result = use_case.execute(
            question=request.message,
            available_domains=request.available_domains or None,
            k=request.k,
        )
        return {"success": True, **_result_to_dict(result)}
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


@router.post("/ask")
def ask(
    request: ChatRequest,
    use_case: QueryRagUseCase = Depends(get_query_rag_use_case),
):
    try:
        result = use_case.execute(
            question=request.message,
            available_domains=request.available_domains or None,
            k=request.k,
        )
        return {"success": True, "data": _result_to_dict(result)}
    except Exception as e:
        logger.error(f"[/ask] Error: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": str(e)}
