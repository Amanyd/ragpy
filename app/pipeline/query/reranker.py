
import logging
import threading

from llama_index.core.postprocessor.types import BaseNodePostprocessor

from app.config.settings import settings

logger = logging.getLogger(__name__)

# Global lock to serialize GPU cross-encoder inference across concurrent asyncio threads
# Prevents PyTorch "RuntimeError: expected scalar type Float but found Half" caused by
# concurrent calls to self.model.half() in FlagEmbedding.
_RERANK_LOCK = threading.Lock()


def get_reranker(top_n: int | None = None) -> BaseNodePostprocessor:
    if top_n is None:
        top_n = settings.reranker_top_n
    try:
        from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
        reranker = FlagEmbeddingReranker(
            model=settings.reranker_model_name,
            top_n=top_n,
            use_fp16=True,
        )

        orig_postprocess = reranker.postprocess_nodes

        def _thread_safe_postprocess(nodes, query_bundle=None):
            if not nodes:
                return []
            if len(nodes) <= (top_n or 6):
                return nodes
            with _RERANK_LOCK:
                try:
                    return orig_postprocess(nodes, query_bundle)
                except Exception as err:
                    logger.warning("reranker_call_failed fallback_to_unranked err=%s", err)
                    return nodes[:top_n]

        reranker.postprocess_nodes = _thread_safe_postprocess
        return reranker
    except Exception as e:
        logger.error("reranker_load_failed error=%s", e)
        raise
