"""Unit tests for app/pipeline/ingest/chunker.py."""


from llama_index.core.schema import Document

from app.pipeline.ingest.chunker import chunk_documents


_LONG_TEXT = " ".join(["word"] * 2000)  # 2000 words far exceeds any default chunk size


def test_long_document_produces_multiple_nodes():
    doc = Document(text=_LONG_TEXT, metadata={"course_id": "c1"})
    nodes = chunk_documents([doc])
    assert len(nodes) > 1


def test_short_document_produces_single_node(sample_documents):
    short_doc = Document(text="Short sentence.", metadata={"course_id": "c1"})
    nodes = chunk_documents([short_doc])
    assert len(nodes) == 1


def test_metadata_propagates_to_child_nodes(sample_documents):
    doc = Document(
        text=_LONG_TEXT,
        metadata={"course_id": "c42", "file_id": "f99", "file_name": "test.pdf"},
    )
    nodes = chunk_documents([doc])
    for node in nodes:
        assert node.metadata.get("course_id") == "c42"
        assert node.metadata.get("file_id") == "f99"
