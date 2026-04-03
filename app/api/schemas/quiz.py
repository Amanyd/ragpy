

from typing import Literal

from pydantic import BaseModel, Field

# Re-export QuizOutput from the pipeline — single source of truth.
from app.pipeline.quiz.formatter import QuizOutput  # noqa: F401


class QuizRequest(BaseModel):

    course_id: str
    difficulty: Literal["easy", "medium", "hard"] = Field(default="medium")
    limit_chunks: int = Field(default=20, ge=5, le=100)

