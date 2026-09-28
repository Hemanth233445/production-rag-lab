"""Small BM25-style keyword retriever for the local reference corpus."""

import math
import re

from app.models import Chunk
from app.retrieval.vector_search import SearchHit


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class KeywordSearch:
    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}

    def add(self, chunks: list[Chunk]) -> None:
        self._chunks.update({chunk.id: chunk for chunk in chunks})

    def search(self, query: str, top_k: int = 8, filters: dict | None = None) -> list[SearchHit]:
        query_terms = set(_tokens(query))
        documents = [
            (chunk, _tokens(chunk.text))
            for chunk in self._chunks.values()
            if all(chunk.metadata.get(key) == value for key, value in (filters or {}).items())
        ]
        if not query_terms or not documents or top_k <= 0:
            return []

        average_length = sum(len(tokens) for _, tokens in documents) / len(documents) or 1.0
        hits = []
        for chunk, tokens in documents:
            length = len(tokens) or 1
            score = 0.0
            for term in query_terms:
                frequency = tokens.count(term)
                if not frequency:
                    continue
                document_frequency = sum(term in doc_tokens for _, doc_tokens in documents)
                inverse_frequency = math.log(1 + (len(documents) - document_frequency + 0.5) / (document_frequency + 0.5))
                denominator = frequency + 1.2 * (1 - 0.75 + 0.75 * length / average_length)
                score += inverse_frequency * frequency * 2.2 / denominator
            if score > 0:
                hits.append(SearchHit(chunk, score))
        return sorted(hits, key=lambda hit: (-hit.score, hit.chunk.id))[:top_k]