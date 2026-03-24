"""Unit tests for app/pipeline/quiz/formatter.py."""


from unittest.mock import AsyncMock, patch

import pytest
from llama_index.core.schema import TextNode

from app.pipeline.quiz.formatter import QuizOutput, format_quiz


@pytest.mark.asyncio
async def test_format_quiz_returns_valid_output(sample_nodes):
    """format_quiz should return a QuizOutput with the correct course_id."""
    fake_output = QuizOutput(
        course_id="course-test",
        questions=[],
    )
    mock_llm = AsyncMock()
    mock_llm.astructured_predict = AsyncMock(return_value=fake_output)

    with patch("app.pipeline.quiz.formatter.get_llm", return_value=mock_llm):
        result = await format_quiz(sample_nodes, course_id="course-test")

    assert isinstance(result, QuizOutput)
    assert result.course_id == "course-test"
    mock_llm.astructured_predict.assert_awaited_once()


@pytest.mark.asyncio
async def test_format_quiz_calls_llm_exactly_once(sample_nodes):
    """LLM structured predict should be called exactly once regardless of node count."""
    fake_output = QuizOutput(course_id="c1", questions=[])
    mock_llm = AsyncMock()
    mock_llm.astructured_predict = AsyncMock(return_value=fake_output)

    with patch("app.pipeline.quiz.formatter.get_llm", return_value=mock_llm):
        await format_quiz(sample_nodes, course_id="c1")

    assert mock_llm.astructured_predict.await_count == 1
