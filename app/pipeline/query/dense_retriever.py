
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters

from app.config.settings import settings
from app.store.qdrant import get_vector_store


def get_retriever(course_id: str, top_k: int | None = None) -> BaseRetriever:
    if top_k is None:
        top_k = settings.retrieval_top_k
    vector_store = get_vector_store()
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    filters = MetadataFilters(
        filters=[MetadataFilter(key="course_id", operator=FilterOperator.EQ, value=course_id)]
    )
    return index.as_retriever(similarity_top_k=top_k, filters=filters)

