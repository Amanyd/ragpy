"""Unit tests for app/pipeline/ingest/parser.py."""


import tempfile
from pathlib import Path

import pytest

from app.pipeline.ingest.parser import parse_file


def _write_temp(suffix: str, content: str) -> Path:
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w")
    f.write(content)
    f.flush()
    f.close()
    return Path(f.name)


def test_parse_txt_returns_documents():
    path = _write_temp(".txt", "Hello world. This is a test document with some content.")
    docs = parse_file(path)
    assert len(docs) > 0
    assert any(len(d.text) > 0 for d in docs)


def test_parse_unsupported_extension_raises():
    path = _write_temp(".xyz", "some content")
    with pytest.raises(ValueError, match="Unsupported"):
        parse_file(path)
