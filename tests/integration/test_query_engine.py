"""Integration test for the query engine against a seeded Qdrant container."""


from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.schema import TextNode
from testcontainers.qdrant import QdrantContainer


@pytest.fixture(scope="module")
def seeded_qdrant():
    """Start Qdrant, seed it with test nodes, return (client, url)."""
    from qdrant_client import AsyncQdrantClient, QdrantClient
    from qdrant_client.http import models as qm

    with QdrantContainer() as container:
        host = container.get_container_host_ip()
        port = int(container.get_exposed_port(6333))
        url = f"http://{host}:{port}"
        client = QdrantClient(url=url)
        aclient = AsyncQdrantClient(url=url)

        collection = "query_test_collection"
        client.create_collection(
            collection_name=collection,
            vectors_config=qm.VectorParams(size=768, distance=qm.Distance.COSINE),
        )

        # Insert a dummy point.
        client.upsert(
            collection_name=collection,
            points=[
                qm.PointStruct(
                    id=1,
                    vector=[0.1] * 768,
                    payload={
                        "course_id": "course-query",
                        "text": "Artificial intelligence enables machines to learn.",
                        "file_name": "ai.pdf",
                        "file_id": "file-q1",
                    },
                )
            ],
        )
        yield client, aclient, collection


@pytest.mark.asyncio
async def test_query_engine_returns_response(seeded_qdrant):
    """Query engine should return non-empty response with correct course scoping."""
    real_client, real_aclient, collection = seeded_qdrant

    from llama_index.core.embeddings.mock_embed_model import MockEmbedding
    from llama_index.core.llms.mock import MockLLM

    fake_embedding = [0.1] * 768
    mock_embed_model = MockEmbedding(embed_dim=768)

    mock_llm = MockLLM()

    with (
        patch("app.store.qdrant._get_client", return_value=real_client),
        patch("app.store.qdrant._get_aclient", return_value=real_aclient),
        patch("app.store.qdrant.settings.qdrant_collection_name", collection),
        patch("app.llm.factory.get_embed_model", return_value=mock_embed_model),
        patch("app.llm.factory.get_llm", return_value=mock_llm),
    ):
        from llama_index.core import Settings as LlamaIndexSettings
        LlamaIndexSettings.llm = mock_llm
        LlamaIndexSettings.embed_model = mock_embed_model

        from app.pipeline.query.engine import get_query_engine

        engine = get_query_engine(course_id="course-query", streaming=False)
        response = await engine.aquery("What does artificial intelligence do?")

    assert response is not None
    assert str(response).strip() != ""


@pytest.mark.asyncio
async def test_query_engine_scopes_to_course(seeded_qdrant):
    """Source nodes in the response must all have the requested course_id."""
    real_client, real_aclient, collection = seeded_qdrant

    from llama_index.core.embeddings.mock_embed_model import MockEmbedding
    from llama_index.core.llms.mock import MockLLM

    fake_embedding = [0.1] * 768
    mock_embed_model = MockEmbedding(embed_dim=768)

    mock_llm = MockLLM()

    with (
        patch("app.store.qdrant._get_client", return_value=real_client),
        patch("app.store.qdrant._get_aclient", return_value=real_aclient),
        patch("app.store.qdrant.settings.qdrant_collection_name", collection),
        patch("app.llm.factory.get_embed_model", return_value=mock_embed_model),
        patch("app.llm.factory.get_llm", return_value=mock_llm),
    ):
        from llama_index.core import Settings as LlamaIndexSettings
        LlamaIndexSettings.llm = mock_llm
        LlamaIndexSettings.embed_model = mock_embed_model

        from app.pipeline.query.engine import get_query_engine

        engine = get_query_engine(course_id="course-query", streaming=False)
        response = await engine.aquery("What does artificial intelligence do?")

    for node in response.source_nodes:
        assert node.metadata.get("course_id") == "course-query"
