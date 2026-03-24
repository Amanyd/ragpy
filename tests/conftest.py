"""Shared pytest fixtures for the RAG microservice test suite."""


from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.schema import Document, TextNode

from app.pipeline.ingest.chunker import chunk_documents


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """Patch all external service URLs to localhost stubs."""
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("MINIO_ENDPOINT", "http://localhost:9000")
    monkeypatch.setenv("NATS_URL", "nats://localhost:4222")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")


@pytest.fixture()
def sample_documents() -> list[Document]:
    """Return a list of Documents with realistic text length."""
    return [
        Document(
            text=(
                "Machine learning is a subset of artificial intelligence that enables "
                "systems to learn and improve from experience without being explicitly "
                "programmed. It focuses on developing computer programs that can access "
                "data and use it to learn for themselves. The learning process begins "
                "with observations or data, such as examples, direct experience, or "
                "instruction, so that computers can look for patterns in data and make "
                "better decisions in the future. The primary aim is to allow computers "
                "to learn automatically without human intervention.\n\n"
                "Deep learning is part of a broader family of machine learning methods "
                "based on artificial neural networks with representation learning. "
                "Learning can be supervised, semi-supervised or unsupervised."
            ),
            metadata={"course_id": "course-1", "file_id": "file-1", "file_name": "ml.pdf"},
        ),
        Document(
            text=(
                "Neural networks are computing systems inspired by the biological neural "
                "networks that constitute animal brains. A neural network consists of "
                "layers of interconnected nodes or neurons that process information using "
                "connectionist approaches to computation. Each neuron receives inputs, "
                "multiplies them by weights, applies an activation function, and passes "
                "the result to the next layer. Training a network involves adjusting "
                "these weights based on the error of predictions compared to true labels, "
                "a process called backpropagation combined with gradient descent."
            ),
            metadata={"course_id": "course-1", "file_id": "file-1", "file_name": "ml.pdf"},
        ),
    ]


@pytest.fixture()
def sample_nodes(sample_documents) -> list[TextNode]:
    """Chunk the sample documents into TextNodes."""
    return chunk_documents(sample_documents)
