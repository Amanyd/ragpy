
from llama_index.core.query_engine import RetrieverQueryEngine

from app.config.settings import settings
from app.pipeline.query.reranker import get_reranker
from app.pipeline.query.synthesizer import get_synthesizer


def get_query_engine(course_id: str, streaming: bool = False) -> RetrieverQueryEngine:
    if settings.hybrid_search_enabled:
        from app.pipeline.query.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever(course_id=course_id)
    else:
        from app.pipeline.query.retriever import get_retriever
        retriever = get_retriever(course_id=course_id)

    reranker = get_reranker()
    synthesizer = get_synthesizer(streaming=streaming)

    return RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[reranker],
        response_synthesizer=synthesizer,
    )

