"""Quiz generation endpoint — async via NATS."""


import json
import logging

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.api.schemas.quiz import QuizRequest
from app.messaging.client import get_nc
from app.messaging.subjects import RAG_QUIZ_PUBLISH_SUBJECT

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/generate", status_code=status.HTTP_202_ACCEPTED, tags=["quiz"])
async def generate_quiz(request: QuizRequest) -> JSONResponse:
    """Enqueue an async quiz generation request. Result delivered via NATS quiz.generate.done."""
    payload = {
        "course_id": request.course_id,
        "difficulty": request.difficulty,
        "limit_chunks": request.limit_chunks,
    }
    nc = get_nc()
    await nc.publish(RAG_QUIZ_PUBLISH_SUBJECT, json.dumps(payload).encode())
    logger.info("quiz_enqueued course_id=%s difficulty=%s", request.course_id, request.difficulty)
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={"status": "queued", "course_id": request.course_id, "difficulty": request.difficulty},
    )
