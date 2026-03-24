"""Integration test for the ingest pipeline using a real Qdrant via testcontainers."""


import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core import Settings as LlamaIndexSettings
from testcontainers.qdrant import QdrantContainer


@pytest.fixture(scope="module")
def qdrant_container():
    """Start a real Qdrant container for the duration of this module."""
    with QdrantContainer() as container:
        yield container



@pytest.mark.asyncio
async def test_run_ingest_indexes_nodes(qdrant_container, tmp_path):
    """Full ingest pipeline stores nodes in Qdrant with correct course_id."""
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qm

    # Resolve the actual URL from the container.
    host = qdrant_container.get_container_host_ip()
    port = int(qdrant_container.get_exposed_port(6333))
    url = f"http://{host}:{port}"

    # Write a small test file.
    doc_path = tmp_path / "sample.txt"
    doc_path.write_text(
        "Artificial intelligence is transforming education. "
        "Students can now access personalised learning experiences. "
        "AI tutors analyse performance and adapt content accordingly."
    )

    from llama_index.core.embeddings.mock_embed_model import MockEmbedding
    mock_embed_model = MockEmbedding(embed_dim=768)

    # Patch the Qdrant singleton to point at our container.
    real_client = QdrantClient(url=url)

    with (
        patch("app.store.qdrant._get_client", return_value=real_client),
        patch("app.store.qdrant.settings.qdrant_collection_name", "test_collection"),
        patch("app.store.qdrant.settings.qdrant_vector_size", 768),
        patch("app.pipeline.ingest.embedder.get_embed_model", return_value=mock_embed_model),
        patch(
            "app.pipeline.ingest.loader.load_file",
            new=AsyncMock(return_value=doc_path),
        ),
    ):
        from llama_index.core import Settings as LlamaIndexSettings
        LlamaIndexSettings.embed_model = mock_embed_model

        from app.store.qdrant import ensure_collection
        from app.pipeline.ingest.pipeline import run_ingest

        await ensure_collection()
        result = await run_ingest(
            bucket="test-bucket",
            key="sample.txt",
            course_id="course-integration",
            file_id="file-int-1",
            file_name="sample.txt",
            teacher_id="teacher-1",
        )

    assert result["status"] == "success"
    assert result["nodes_indexed"] > 0

    # Verify the nodes are in Qdrant with the correct course_id payload.
    records, _ = real_client.scroll(
        collection_name="test_collection",
        scroll_filter=qm.Filter(
            must=[qm.FieldCondition(key="course_id", match=qm.MatchValue(value="course-integration"))]
        ),
        with_payload=True,
        limit=100,
    )
    assert len(records) > 0
    for record in records:
        assert record.payload.get("course_id") == "course-integration"
