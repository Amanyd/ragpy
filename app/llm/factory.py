
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.ollama import Ollama

from app.config.settings import settings

_llm: Ollama | None = None

def get_llm() -> Ollama:
    global _llm
    if _llm is None:
        _llm = Ollama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            request_timeout=600.0,
            additional_kwargs={"num_ctx": 4096}
        )
    return _llm


_embed_model: FastEmbedEmbedding | None = None

def get_embed_model() -> FastEmbedEmbedding:
    global _embed_model
    if _embed_model is None:
        _embed_model = FastEmbedEmbedding(model_name=settings.embed_model_name)
    return _embed_model

