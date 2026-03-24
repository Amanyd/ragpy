

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):

    course_id: str
    query: str = Field(min_length=1, max_length=2000)
    stream: bool = True


class CitationItem(BaseModel):

    file_name: str
    file_id: str
    score: float | None = None


class ChatResponse(BaseModel):

    answer: str
    citations: list[CitationItem]
