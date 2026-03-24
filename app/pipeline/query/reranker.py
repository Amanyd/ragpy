
import logging

from llama_index.core.postprocessor.types import BaseNodePostprocessor

logger = logging.getLogger(__name__)


def get_reranker(top_n: int = 3) -> BaseNodePostprocessor:
    try:
        from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
        from app.config.settings import settings
        return FlagEmbeddingReranker(
            model=settings.reranker_model_name,
            top_n=top_n,
            use_fp16=True,
        )
    except Exception as e:
        logger.error("reranker_load_failed error=%s", e)
        raise
