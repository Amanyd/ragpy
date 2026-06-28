"""Semantic grading for open-ended quiz answers via LLM judge."""

import logging

from pydantic import BaseModel, Field

from app.llm.factory import get_llm
from app.llm.prompts import QUIZ_GRADING_PROMPT
from app.llm.structured import astructured_predict_json

logger = logging.getLogger(__name__)


class GradeResult(BaseModel):
    """Structured grading output from the LLM judge."""

    is_correct: bool
    score: float = Field(ge=0.0, le=1.0)
    explanation: str = ""


async def grade_answer(question: str, reference_answer: str, user_answer: str) -> GradeResult:
    """Grade a student's open-ended answer semantically using the LLM judge."""
    llm = get_llm()
    result: GradeResult = await astructured_predict_json(
        llm,
        GradeResult,
        QUIZ_GRADING_PROMPT,
        question=question,
        reference_answer=reference_answer,
        user_answer=user_answer,
    )
    logger.info(
        "graded question=%s is_correct=%s score=%.2f",
        question[:60],
        result.is_correct,
        result.score,
    )
    return result
