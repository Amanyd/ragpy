"""BM25 (sparse keyword) retriever over Qdrant-stored nodes.

Fetches all text nodes for a course from Qdrant, tokenises them,
and scores against the query using BM25Okapi.
"""

import logging
import re
from functools import lru_cache

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from qdrant_client.http import models as qmodels
from rank_bm25 import BM25Okapi

from app.config.settings import settings
from app.store.qdrant import get_sync_client

logger = logging.getLogger(__name__)

_SPLIT_RE = re.compile(r"\W+")


def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokenizer — intentionally simple."""
    return [t for t in _SPLIT_RE.split(text.lower()) if t]


def _fetch_course_nodes(course_id: str) -> list[TextNode]:
    """Scroll through Qdrant to fetch all nodes for a course."""
    client = get_sync_client()
    collection = settings.qdrant_collection_name
    nodes: list[TextNode] = []
    offset = None

    while True:
        results, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="course_id",
                        match=qmodels.MatchValue(value=course_id),
                    )
                ]
            ),
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in results:
            payload = point.payload or {}
            text = payload.get("_node_content", "")
            # LlamaIndex stores serialised JSON in _node_content; fall back to text field
            if not text:
                text = payload.get("text", "")
            # Try to extract just the text from the JSON blob
            if text.startswith("{"):
                try:
                    import json
                    text = json.loads(text).get("text", text)
                except Exception:
                    pass
            metadata = {k: v for k, v in payload.items() if k not in ("_node_content", "text")}
            node = TextNode(text=text, id_=str(point.id), metadata=metadata)
            nodes.append(node)

        if next_offset is None:
            break
        offset = next_offset

    logger.info("bm25_fetched course_id=%s nodes=%d", course_id, len(nodes))
    return nodes


# Simple cache keyed on course_id — avoids re-fetching on every query.
# For production, add a TTL or invalidation on ingest.
@lru_cache(maxsize=32)
def _build_bm25_index(course_id: str) -> tuple[BM25Okapi, list[TextNode]]:
    nodes = _fetch_course_nodes(course_id)
    if not nodes:
        return BM25Okapi([["__empty__"]]), []
    corpus = [_tokenize(n.text) for n in nodes]
    return BM25Okapi(corpus), nodes


def bm25_retrieve(query: str, course_id: str, top_k: int | None = None) -> list[NodeWithScore]:
    """Return top_k nodes scored by BM25 for the given query and course."""
    if top_k is None:
        top_k = settings.retrieval_top_k

    bm25, nodes = _build_bm25_index(course_id)
    if not nodes:
        return []

    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # Pair (score, node) and sort descending
    scored = sorted(zip(scores, nodes), key=lambda x: x[0], reverse=True)

    results: list[NodeWithScore] = []
    for score, node in scored[:top_k]:
        if score <= 0:
            break
        results.append(NodeWithScore(node=node, score=float(score)))

    logger.debug("bm25_retrieve course=%s query=%s results=%d", course_id, query[:60], len(results))
    return results
