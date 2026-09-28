"""Word-window chunking with stable IDs and metadata propagation."""

import hashlib

from app.models import Chunk, Document


def chunk_document(
    document: Document,
    chunk_size: int = 180,
    overlap: int = 30,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    words = document.text.split()
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        text = " ".join(words[start : start + chunk_size]).strip()
        if not text:
            continue
        index = len(chunks)
        source = str(document.metadata.get("source", "document"))
        digest = hashlib.sha256(f"{source}:{index}:{text}".encode("utf-8")).hexdigest()[:16]
        metadata = {**document.metadata, "chunk_index": index}
        chunks.append(Chunk(id=digest, text=text, metadata=metadata))
        if start + chunk_size >= len(words):
            break
    return chunks