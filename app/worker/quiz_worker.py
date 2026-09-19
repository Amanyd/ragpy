"""NATS JetStream pull consumer for async quiz generation."""


import asyncio
import json
import logging

import nats.errors
from nats.aio.msg import Msg
from nats.js.api import ConsumerConfig

from app.messaging.client import get_js
from app.messaging.subjects import (
    DURABLE_QUIZ_WORKER,
    RAG_QUIZ_DONE_SUBJECT,
    RAG_QUIZ_PUBLISH_SUBJECT,
    RAG_QUIZ_STREAM,
)
from app.pipeline.quiz.pipeline import generate_course_quiz

logger = logging.getLogger(__name__)


async def process_quiz_message(msg: Msg, sem: asyncio.Semaphore) -> None:
    """Parse, validate, and process a single quiz generation message."""
    async with sem:
        try:
            payload = json.loads(msg.data.decode())
        except Exception:
            logger.exception("quiz_msg invalid json subject=%s", msg.subject)
            await msg.term()
            return

        quiz_type: str | None = payload.get("type")
        course_id: str | None = payload.get("course_id")
        lesson_id: str | None = payload.get("lesson_id")
        file_id: str | None = payload.get("file_id")
        keywords: list[str] = payload.get("keywords", [])
        
        if not course_id or not quiz_type:
            logger.error("quiz_msg missing required fields subject=%s", msg.subject)
            await msg.term()
            return

        difficulty: str = payload.get("difficulty", "medium")
        limit_chunks: int = int(payload.get("limit_chunks", 20))

        js = get_js()

        try:
            # Reusing generate_course_quiz name but updating its internals in pipeline.py
            result, extracted_keywords = await generate_course_quiz(
                quiz_type=quiz_type,
                course_id=course_id,
                lesson_id=lesson_id,
                file_id=file_id,
                keywords=keywords,
                difficulty=difficulty,
                limit_chunks=limit_chunks,
            )

            done_payload = {
                "type": quiz_type,
                "status": "success",
                "course_id": course_id,
                "lesson_id": lesson_id,
                "difficulty": difficulty,
                "keywords": extracted_keywords,
                "questions": json.loads(result.model_dump_json())["questions"],
            }
            await js.publish(RAG_QUIZ_DONE_SUBJECT, json.dumps(done_payload).encode())
            logger.info("quiz_done type=%s course_id=%s difficulty=%s questions=%d", quiz_type, course_id, difficulty, len(result.questions))
            await msg.ack()

        except Exception:
            logger.exception("quiz_failed type=%s course_id=%s difficulty=%s", quiz_type, course_id, difficulty)
            done_payload = {
                "type": quiz_type,
                "status": "failed",
                "course_id": course_id,
                "lesson_id": lesson_id,
                "difficulty": difficulty,
                "questions": [],
            }
            await js.publish(RAG_QUIZ_DONE_SUBJECT, json.dumps(done_payload).encode())
            # Acknowledge the message since we've permanently failed and notified the backend
            await msg.ack()


async def start_quiz_worker() -> None:
    """Start a pull consumer loop for async quiz generation."""
    js = get_js()
    psub = await js.pull_subscribe(
        RAG_QUIZ_PUBLISH_SUBJECT,
        durable=DURABLE_QUIZ_WORKER,
        stream=RAG_QUIZ_STREAM,
        config=ConsumerConfig(max_deliver=5),
    )
    logger.info("quiz_worker pull_subscribe registered subject=%s", RAG_QUIZ_PUBLISH_SUBJECT)

    sem = asyncio.Semaphore(3)

    while True:
        try:
            # Fetch up to 3 messages at once
            msgs = await psub.fetch(batch=3, timeout=1.0)
            for msg in msgs:
                asyncio.create_task(process_quiz_message(msg, sem))
        except nats.errors.TimeoutError:
            continue
        except Exception as e:
            logger.error("quiz_worker fetch loop error e=%s", e)
            await asyncio.sleep(1)
