"""Pydantic output models and structured quiz formatting from node context."""


import logging
from typing import Literal

from llama_index.core.schema import BaseNode
from pydantic import BaseModel

from app.llm.factory import get_llm
from app.llm.prompts import QUIZ_GENERATION_PROMPT

logger = logging.getLogger(__name__)


class QuizChoice(BaseModel):
    """A single answer choice for an MCQ question."""

    label: str
    text: str


class QuizQuestion(BaseModel):
    """A single quiz question, either MCQ or open-ended."""

    type: Literal["mcq", "open_ended"]
    question: str
    choices: list[QuizChoice] | None = None
    answer: str


class QuizOutput(BaseModel):
    """Structured quiz output for a course."""

    course_id: str
    questions: list[QuizQuestion]


async def format_quiz(nodes: list[BaseNode], course_id: str) -> QuizOutput:
    """Call LLM with structured prediction to produce a QuizOutput from nodes."""
    context_parts: list[str] = []
    for node in nodes:
        text = node.text or ""
        qa_meta = node.metadata.get("questions_this_excerpt_can_answer", "")
        if text:
            context_parts.append(text)
        if qa_meta:
            context_parts.append(f"Q&A hints: {qa_meta}")

    context_str = "\n\n---\n\n".join(context_parts)

    llm = get_llm()
    # astructured_predict injects schema automatically; prompt_args fill template vars.
    result: QuizOutput = await llm.astructured_predict(
        QuizOutput,
        QUIZ_GENERATION_PROMPT,
        context_str=context_str,
    )

    # Stamp course_id on the result (LLM may not know it).
    result.course_id = course_id

    logger.info("quiz formatted course_id=%s questions=%d", course_id, len(result.questions))
    return result
