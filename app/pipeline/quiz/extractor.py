"""Extract Q&A pairs from document nodes using QuestionsAnsweredExtractor."""


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


from llama_index.core.prompts.base import PromptTemplate
from pydantic import BaseModel

from app.llm.structured import astructured_predict_json


class KeywordExtraction(BaseModel):
    keywords: list[str]

async def extract_keywords_from_nodes(nodes: list[BaseNode]) -> list[str]:
    """Extract syllabus keywords from lesson plan nodes."""
    if not nodes:
        return []

    context_parts = [node.text for node in nodes if node.text]
    context_str = "\n\n---\n\n".join(context_parts)
    
    # Truncate to roughly ~10,000 tokens (40,000 chars) to comfortably fit in the 32k context limit
    if len(context_str) > 40000:
        context_str = context_str[:40000] + "\n...[TRUNCATED]"
    
    prompt = PromptTemplate(
        "You are a curriculum expert. Read the following lesson plan. "
        "Ignore administrative details (creator, date, boilerplate). "
        "Extract ONLY the core educational topics, learning objectives, and key concepts.\n"
        "Output the result as a valid JSON object with a single key \"keywords\" containing a list of strings.\n"
        "Example: {{\"keywords\": [\"keyword 1\", \"keyword 2\"]}}\n\n"
        "Lesson Plan:\n{context_str}"
    )
    
    try:
        llm = get_llm()
        result = await astructured_predict_json(
            llm,
            KeywordExtraction,
            prompt,
            context_str=context_str,
        )
        logger.info("keywords_extracted count=%d", len(result.keywords))
        return result.keywords
    except Exception as e:
        logger.error("failed to extract keywords: %s", e)
        return []
