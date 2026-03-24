

from typing import Literal

from pydantic import BaseModel


class IngestRequest(BaseModel):

    bucket: str
    key: str
    course_id: str
    file_id: str
    file_name: str
    teacher_id: str


class IngestResponse(BaseModel):

    status: Literal["queued"]
    file_id: str
