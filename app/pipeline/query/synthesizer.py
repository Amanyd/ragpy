from typing import Any

from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.response_synthesizers.base import BaseSynthesizer

from app.llm.prompts import QA_PROMPT


def get_synthesizer(streaming: bool = False) -> BaseSynthesizer:
    return get_response_synthesizer(
        response_mode=ResponseMode.COMPACT,
        streaming=streaming,
        text_qa_template=QA_PROMPT,
    )


def format_citations(response: Any) -> list[dict]:
    items: list[dict] = []
    source_nodes = response.source_nodes or []
    for node_with_score in source_nodes:
        node = node_with_score.node
        score = node_with_score.score
        metadata = node.metadata or {}
        file_name = metadata.get("file_name")
        file_id = metadata.get("file_id")
        if not file_name or not file_id:
            continue
        items.append({"file_name": file_name, "file_id": file_id, "score": score})
    return items
