"""Generate a structured quiz for a course by scrolling stored Qdrant nodes."""


import asyncio
import json
import logging

from llama_index.core.schema import TextNode
from qdrant_client.http import models as qdrant_models

from app.pipeline.quiz.extractor import extract_qa_pairs
from app.pipeline.quiz.formatter import QuizOutput, format_quiz
from app.store.qdrant import get_sync_client
from app.config.settings import settings

logger = logging.getLogger(__name__)


async def generate_course_quiz(
    course_id: str,
    limit_chunks: int = 20,
) -> QuizOutput:
    """Fetch course nodes from Qdrant, extract Q&A pairs, and format a quiz."""
    client = get_sync_client()
    collection_name = settings.qdrant_collection_name

    def _scroll() -> list[TextNode]:
        scroll_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="course_id",
                    match=qdrant_models.MatchValue(value=course_id),
                )
            ]
        )
        records, _next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            with_payload=True,
            with_vectors=False,
            limit=limit_chunks,
        )
        nodes: list[TextNode] = []
        for record in records:
            payload = record.payload or {}
            text = payload.get("text", "")
            
            if not text and "_node_content" in payload:
                try:
                    node_content = json.loads(payload["_node_content"])
                    text = node_content.get("text", "")
                except json.JSONDecodeError:
                    pass
            
            metadata = {k: v for k, v in payload.items() if k not in ("text", "_node_content")}
            nodes.append(TextNode(id_=str(record.id), text=text, metadata=metadata))
        return nodes

    logger.info("quiz_generate course_id=%s limit=%d", course_id, limit_chunks)
    nodes = await asyncio.to_thread(_scroll)

    if not nodes:
        logger.warning("no nodes found course_id=%s", course_id)
        return QuizOutput(course_id=course_id, questions=[])

    enriched_nodes = await extract_qa_pairs(nodes)
    result = await format_quiz(enriched_nodes, course_id)
    logger.info("quiz_done course_id=%s questions=%d", course_id, len(result.questions))
    return result
