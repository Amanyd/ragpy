"""Chat endpoint — supports both streaming SSE and non-streaming JSON."""


import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.schemas.chat import ChatRequest, ChatResponse, CitationItem
from app.pipeline.query.engine import get_query_engine
from app.pipeline.query.synthesizer import format_citations

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/", tags=["chat"])
async def chat(request: ChatRequest):
    """Answer a student query; streaming or non-streaming based on request.stream."""
    engine = get_query_engine(course_id=request.course_id, streaming=request.stream)

    if request.stream:
        return StreamingResponse(
            _sse_generator(engine, request.query),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    # Non-streaming: call async aquery and return JSON.
    response = await engine.aquery(request.query)
    citations = format_citations(response)
    return ChatResponse(
        answer=str(response),
        citations=[CitationItem(**c) for c in citations],
    )


async def _sse_generator(engine, query: str):
    """Async generator yielding SSE frames for a streaming LlamaIndex query."""
    try:
        # LlamaIndex async streaming query
        streaming_response = await engine.aquery(query)

        if not hasattr(streaming_response, "async_response_gen"):
            yield f"data: {json.dumps({'error': 'attempted SSE stream on non-streaming response'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Yield text chunks as SSE data frames.
        async for token in streaming_response.async_response_gen():
            frame = json.dumps({"token": token})
            yield f"data: {frame}\n\n"

        # Yield citations as a final structured frame.
        citations = format_citations(streaming_response)
        yield f"data: {json.dumps({'citations': citations})}\n\n"
        yield "data: [DONE]\n\n"

    except Exception:
        logger.exception("chat_stream_error query=%s", query[:80])
        yield f"data: {json.dumps({'error': 'stream failed'})}\n\n"
        yield "data: [DONE]\n\n"
