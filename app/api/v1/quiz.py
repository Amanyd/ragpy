"""Quiz generation endpoint."""


import logging

from fastapi import APIRouter, HTTPException, status

from app.api.schemas.quiz import QuizOutput, QuizRequest
from app.pipeline.quiz.pipeline import generate_course_quiz

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/generate", response_model=QuizOutput, tags=["quiz"])
async def generate_quiz(request: QuizRequest) -> QuizOutput:
    """Generate a structured quiz for a course from its indexed content."""
    result = await generate_course_quiz(
        course_id=request.course_id,
        limit_chunks=request.limit_chunks,
    )
    if not result.questions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No content indexed for course_id={request.course_id!r}",
        )
    return result
