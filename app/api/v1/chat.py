"""Chat endpoint — supports both streaming SSE and non-streaming JSON."""


import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.schemas.chat import ChatRequest, ChatResponse, CitationItem
from app.pipeline.query.condenser import condense_query
from app.pipeline.query.engine import get_query_engine
from app.pipeline.query.synthesizer import format_citations

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/", tags=["chat"])
async def chat(request: ChatRequest):
    """Answer a student query; streaming or non-streaming based on request.stream."""

    # Condense follow-up questions using conversation history (skip on first message)
    if request.history:
        history_dicts = [{"role": m.role, "content": m.content} for m in request.history]
        query = await asyncio.to_thread(condense_query, request.query, history_dicts)
    else:
        query = request.query

    engine = get_query_engine(course_ids=request.course_ids, streaming=request.stream)

    if request.stream:
        return StreamingResponse(
            _sse_generator(engine, query),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    # Non-streaming: call async aquery and return JSON.
    response = await engine.aquery(query)
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
        logger.info("chat_sse response_type=%s has_async_gen=%s has_sync_gen=%s", 
                    type(streaming_response).__name__,
                    hasattr(streaming_response, "async_response_gen"),
                    hasattr(streaming_response, "response_gen"))

        if hasattr(streaming_response, "async_response_gen"):
            # Yield text chunks as SSE data frames.
            token_count = 0
            async for token in streaming_response.async_response_gen():
                token_count += 1
                frame = json.dumps({"token": token})
                yield f"data: {frame}\n\n"
            logger.info("chat_sse tokens_yielded=%d", token_count)
        elif hasattr(streaming_response, "response_gen"):
            # Fallback: some LlamaIndex versions use sync generator
            token_count = 0
            for token in streaming_response.response_gen:
                token_count += 1
                frame = json.dumps({"token": token})
                yield f"data: {frame}\n\n"
            logger.info("chat_sse tokens_yielded_sync=%d", token_count)
        else:
            yield f"data: {json.dumps({'error': 'attempted SSE stream on non-streaming response'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Yield citations as a final structured frame.
        citations = format_citations(streaming_response)
        yield f"data: {json.dumps({'citations': citations})}\n\n"
        yield "data: [DONE]\n\n"

    except Exception:
        logger.exception("chat_stream_error query=%s", query[:80])
        yield f"data: {json.dumps({'error': 'stream failed'})}\n\n"
        yield "data: [DONE]\n\n"

