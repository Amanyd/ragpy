"""Unit tests for app/pipeline/ingest/embedder.py."""


from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core.schema import TextNode


@pytest.mark.asyncio
async def test_embed_nodes_populates_embeddings(sample_nodes):
    """Embedder should attach a float-list embedding to every node."""
    fake_embedding = [0.1] * 768

    from llama_index.core.embeddings.mock_embed_model import MockEmbedding
    
    # Mock the embed model so no real model is loaded.
    mock_embed_model = MockEmbedding(embed_dim=768)

    with patch("app.pipeline.ingest.embedder.get_embed_model", return_value=mock_embed_model):
        from app.pipeline.ingest.embedder import embed_nodes

        result = await embed_nodes(sample_nodes)

    assert len(result) == len(sample_nodes)
    for node in result:
        assert node.embedding is not None
        assert len(node.embedding) == 768
