import pytest

from app.ingestion.chunker import chunk_document
from app.models import Document


def test_chunking_preserves_metadata_and_stable_ids():
    document = Document("one two three four five six", {"source": "guide.md", "team": "docs"})
    first = chunk_document(document, chunk_size=4, overlap=1)
    second = chunk_document(document, chunk_size=4, overlap=1)
    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert first[0].metadata["team"] == "docs"
    assert first[1].text.startswith("four")


def test_chunking_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_document(Document("text"), chunk_size=4, overlap=4)