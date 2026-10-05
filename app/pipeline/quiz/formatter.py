"""Pydantic output models and structured quiz formatting from node context."""


import logging
from typing import Literal

from llama_index.core.schema import BaseNode
from pydantic import BaseModel, model_validator

from app.llm.factory import get_llm
from app.llm.prompts import QUIZ_GENERATION_PROMPT, TOPIC_SYNTHESIS_PROMPT
from app.llm.structured import astructured_predict_json

logger = logging.getLogger(__name__)


class QuizChoice(BaseModel):
    """A single answer choice for an MCQ question."""

    label: str
    text: str


class QuizQuestion(BaseModel):
    """A single quiz question, either MCQ or open-ended."""

    type: Literal["mcq", "open_ended"] = "mcq"
    question: str
    choices: list[QuizChoice] | None = None
    answer: str
    explanation: str | None = None
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    topic_phrase: str | None = None

    @model_validator(mode="after")
    def validate_choices(self) -> "QuizQuestion":
        if self.type == "mcq":
            if not self.choices or len(self.choices) < 2:
                raise ValueError("MCQ question must have at least 2 choices")
        elif self.type == "open_ended":
            pass
        return self


class TopicSlide(BaseModel):
    """A single micro-learning educational slide."""

    slide_number: int
    slide_type: Literal["concept", "technical_limits", "diagram", "emergency"]
    title: str
    bullets: list[str] = []
    formula_or_rule: str | None = None
    diagram_mermaid: str | None = None
    warning: str | None = None


class TopicSynthesisOutput(BaseModel):
    """Combined output for a topic: 4 slides + 6 questions."""

    slides: list[TopicSlide] = []
    questions: list[QuizQuestion] = []


class TopicSummary(BaseModel):
    """Summary of a topic with its slides."""

    title: str
    slides: list[TopicSlide] = []


class QuizOutput(BaseModel):
    """Structured quiz output for a course or lesson."""

    course_id: str = ""
    lesson_id: str | None = None
    questions: list[QuizQuestion]
    topics: list[TopicSummary] = []


async def synthesize_topic(
    nodes: list[BaseNode],
    topic_phrase: str,
) -> TopicSynthesisOutput:
    """Generate 4 micro-learning slides + 6 questions for a specific topic."""
    context_parts: list[str] = []
    for node in nodes:
        text = node.text or ""
        if text:
            context_parts.append(text)

    context_str = "\n\n---\n\n".join(context_parts)

    llm = get_llm()
    try:
        result: TopicSynthesisOutput = await astructured_predict_json(
            llm,
            TopicSynthesisOutput,
            TOPIC_SYNTHESIS_PROMPT,
            context_str=context_str,
            topic_phrase=topic_phrase,
        )
        # Tag each question with its topic phrase
        for q in result.questions:
            q.topic_phrase = topic_phrase
        logger.info(
            "topic_synthesized topic=%s slides=%d questions=%d",
            topic_phrase,
            len(result.slides),
            len(result.questions),
        )
        return result
    except Exception as e:
        logger.error("failed_to_synthesize_topic topic=%s err=%s", topic_phrase, e)
        return TopicSynthesisOutput(slides=[], questions=[])



async def format_quiz(nodes: list[BaseNode], course_id: str, difficulty: str = "medium", quiz_type: str = "course") -> QuizOutput:
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

    if quiz_type == "lesson":
        requirements = (
            "- Generate Exactly 5 questions total.\n"
            "- All 5 questions must be multiple-choice (type: \"mcq\"). Do not generate any open_ended questions.\n"
        )
    else:
        requirements = (
            f"Difficulty level: {difficulty}\n"
            "- easy: Basic recall and definition questions. Straightforward single-concept answers.\n"
            "- medium: Application and understanding questions. May require connecting two concepts.\n"
            "- hard: Analysis and synthesis questions. Requires deep understanding and multi-step reasoning.\n"
            "- Generate Exactly 10 questions total.\n"
            "- 8 must be multiple-choice (type: \"mcq\") and 2 open-ended (type: \"open_ended\").\n"
        )

    llm = get_llm()
    # sglang doesn't return OpenAI tool_calls, so we use JSON-constrained
    # decoding (response_format json_object) + Pydantic validation instead of
    # astructured_predict (which uses function calling and fails here).
    result: QuizOutput = await astructured_predict_json(
        llm,
        QuizOutput,
        QUIZ_GENERATION_PROMPT,
        context_str=context_str,
        requirements=requirements,
    )

    # Stamp course_id on the result (LLM may not know it).
    result.course_id = course_id

    logger.info("quiz formatted course_id=%s questions=%d", course_id, len(result.questions))
    return result
