
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import TextNode

from app.llm.factory import get_embed_model


async def embed_nodes(nodes: list[TextNode]) -> list[TextNode]:
    pipeline = IngestionPipeline(transformations=[get_embed_model()])
    embedded = await pipeline.arun(nodes=nodes, show_progress=False)
    return list(embedded)  # type: ignore[return-value]

