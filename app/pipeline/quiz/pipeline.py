"""Generate a structured quiz for a course by scrolling stored Qdrant nodes.

Scrolls ALL chunks for the course, then stratified-samples across files
so every file gets representation in the quiz.
"""


import asyncio
import json
import logging
import random
from collections import defaultdict

from llama_index.core.schema import TextNode
from qdrant_client.http import models as qdrant_models

from app.pipeline.quiz.extractor import extract_qa_pairs
from app.pipeline.quiz.formatter import QuizOutput, format_quiz
from app.store.qdrant import get_sync_client
from app.config.settings import settings

logger = logging.getLogger(__name__)


def _parse_node(record) -> TextNode:
    """Convert a Qdrant scroll record into a TextNode."""
    payload = record.payload or {}
    text = payload.get("text", "")

    if not text and "_node_content" in payload:
        try:
            node_content = json.loads(payload["_node_content"])
            text = node_content.get("text", "")
        except json.JSONDecodeError:
            pass

    metadata = {k: v for k, v in payload.items() if k not in ("text", "_node_content")}
    return TextNode(id_=str(record.id), text=text, metadata=metadata)


def _scroll_all_nodes(course_id: str) -> list[TextNode]:
    """Paginate through ALL chunks for a course in Qdrant."""
    client = get_sync_client()
    collection_name = settings.qdrant_collection_name
    scroll_filter = qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="course_id",
                match=qdrant_models.MatchValue(value=course_id),
            )
        ]
    )

    all_nodes: list[TextNode] = []
    offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            with_payload=True,
            with_vectors=False,
            limit=256,
            offset=offset,
        )
        all_nodes.extend(_parse_node(r) for r in records)

        if next_offset is None:
            break
        offset = next_offset

    logger.info("quiz_scroll course_id=%s total_chunks=%d", course_id, len(all_nodes))
    return all_nodes


def _stratified_sample(nodes: list[TextNode], budget: int) -> list[TextNode]:
    """Sample `budget` nodes with equal representation from every file.

    Groups chunks by file_name, then does round-robin sampling so each file
    contributes roughly the same number of chunks.
    """
    # Group by file
    by_file: dict[str, list[TextNode]] = defaultdict(list)
    for node in nodes:
        key = node.metadata.get("file_name", "__unknown__")
        by_file[key].append(node)

    file_keys = sorted(by_file.keys())
    num_files = len(file_keys)
    if num_files == 0:
        return []

    # Shuffle within each file so we pick diverse chunks (not just the first N)
    for key in file_keys:
        random.shuffle(by_file[key])

    # Round-robin: deal chunks per file up to budget
    per_file = max(1, budget // num_files)
    sampled: list[TextNode] = []

    for key in file_keys:
        sampled.extend(by_file[key][:per_file])

    # If we still have budget left (rounding), fill from remaining chunks
    if len(sampled) < budget:
        used_ids = {n.node_id for n in sampled}
        remaining = [n for n in nodes if n.node_id not in used_ids]
        random.shuffle(remaining)
        sampled.extend(remaining[: budget - len(sampled)])

    logger.info(
        "quiz_sample files=%d per_file=%d sampled=%d",
        num_files, per_file, len(sampled),
    )
    return sampled[:budget]


async def generate_course_quiz(
    course_id: str,
    difficulty: str = "medium",
    limit_chunks: int = 20,
) -> QuizOutput:
    """Fetch ALL course nodes, stratified-sample across files, generate quiz."""
    logger.info("quiz_generate course_id=%s limit=%d", course_id, limit_chunks)

    all_nodes = await asyncio.to_thread(_scroll_all_nodes, course_id)

    if not all_nodes:
        logger.warning("no nodes found course_id=%s", course_id)
        return QuizOutput(course_id=course_id, questions=[])

    # Stratified sample so every file is represented
    sampled_nodes = _stratified_sample(all_nodes, budget=limit_chunks)

    enriched_nodes = await extract_qa_pairs(sampled_nodes)
    result = await format_quiz(enriched_nodes, course_id, difficulty=difficulty)
    logger.info("quiz_done course_id=%s questions=%d", course_id, len(result.questions))
    return result
