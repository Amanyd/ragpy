"""NATS JetStream push consumer for async quiz generation."""


import asyncio
import json
import logging
import nats.errors

from nats.aio.msg import Msg

from app.messaging.client import get_js, get_nc
from app.messaging.subjects import (
    DURABLE_QUIZ_WORKER,
    RAG_QUIZ_PUBLISH_SUBJECT,
    RAG_QUIZ_STREAM,
)
from app.pipeline.quiz.pipeline import generate_course_quiz

logger = logging.getLogger(__name__)


async def process_quiz_message(msg: Msg) -> None:
    """Parse, validate, and process a single quiz generation message."""
    try:
        payload = json.loads(msg.data.decode())
    except Exception:
        logger.exception("quiz_msg invalid json subject=%s", msg.subject)
        await msg.term()
        return

    course_id: str | None = payload.get("course_id")
    if not course_id:
        logger.error("quiz_msg missing course_id subject=%s", msg.subject)
        await msg.term()
        return

    limit_chunks: int = int(payload.get("limit_chunks", 20))

    try:
        result = await generate_course_quiz(course_id=course_id, limit_chunks=limit_chunks)

        # If requester set a reply subject, publish the result back.
        if msg.reply:
            nc = get_nc()
            await nc.publish(msg.reply, result.model_dump_json().encode())

        await msg.ack()
    except Exception:
        logger.exception("quiz_failed course_id=%s", course_id)
        await msg.nak(delay=5)


async def start_quiz_worker() -> None:
    """Start a pull consumer loop for async quiz generation."""
    js = get_js()
    psub = await js.pull_subscribe(
        RAG_QUIZ_PUBLISH_SUBJECT,
        durable=DURABLE_QUIZ_WORKER,
        stream=RAG_QUIZ_STREAM,
    )
    logger.info("quiz_worker pull_subscribe registered subject=%s", RAG_QUIZ_PUBLISH_SUBJECT)

    while True:
        try:
            msgs = await psub.fetch(batch=5, timeout=1.0)
            for msg in msgs:
                await process_quiz_message(msg)
        except nats.errors.TimeoutError:
            continue
        except Exception as e:
            logger.error("quiz_worker fetch loop error e=%s", e)
            await asyncio.sleep(1)
