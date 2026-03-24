"""Extract Q&A pairs from document nodes using QuestionsAnsweredExtractor."""


import asyncio
import logging


from llama_index.core.extractors import QuestionsAnsweredExtractor
from llama_index.core.schema import BaseNode

from app.llm.factory import get_llm

logger = logging.getLogger(__name__)


async def extract_qa_pairs(
    nodes: list[BaseNode],
    questions_per_chunk: int = 3,
) -> list[BaseNode]:
    """Run QA extraction on nodes; return nodes with enriched metadata."""
    if not nodes:
        return nodes

    extractor = QuestionsAnsweredExtractor(
        llm=get_llm(),
        questions=questions_per_chunk,
    )

    # aextract returns list[dict] — one dict per node with extracted metadata.
    metadata_list: list[dict] = await extractor.aextract(nodes)

    # Merge extracted metadata back into the node objects.
    for node, metadata in zip(nodes, metadata_list):
        node.metadata.update(metadata)

    logger.info("qa_extraction nodes=%d questions_each=%d", len(nodes), questions_per_chunk)
    return nodes
