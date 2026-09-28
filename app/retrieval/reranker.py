"""Lightweight lexical reranker; replace with a cross-encoder when needed."""

import re

from app.retrieval.vector_search import SearchHit


def rerank(query: str, hits: list[SearchHit], top_k: int = 4) -> list[SearchHit]:
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    rescored = []
    for hit in hits:
        text_terms = set(re.findall(r"[a-z0-9]+", hit.chunk.text.lower()))
        overlap = len(query_terms & text_terms) / max(len(query_terms), 1)
        rescored.append(SearchHit(hit.chunk, overlap + hit.score * 0.01))
    return sorted(rescored, key=lambda hit: (-hit.score, hit.chunk.id))[:max(0, top_k)]