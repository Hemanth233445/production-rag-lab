"""Reject answers that cite evidence outside the retrieved context."""

import re

from app.retrieval.vector_search import SearchHit

_CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9_-]+)\]")
_ABSTENTION = "I don't have enough information in the indexed sources to answer that."


def validate_grounding(answer: str, hits: list[SearchHit]) -> str:
    allowed_ids = {hit.chunk.id for hit in hits}
    cited_ids = _CITATION_PATTERN.findall(answer)
    if not cited_ids or not set(cited_ids).issubset(allowed_ids):
        return _ABSTENTION
    return answer