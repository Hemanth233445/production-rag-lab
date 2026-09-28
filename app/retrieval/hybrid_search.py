"""Reciprocal-rank fusion across vector and keyword retrieval."""

from app.retrieval.keyword_search import KeywordSearch
from app.retrieval.vector_search import InMemoryVectorIndex, SearchHit


class HybridSearch:
    def __init__(self, vector_search: InMemoryVectorIndex, keyword_search: KeywordSearch) -> None:
        self.vector_search = vector_search
        self.keyword_search = keyword_search

    def search(
        self,
        query: str,
        top_k: int = 8,
        filters: dict | None = None,
        rank_constant: int = 60,
    ) -> list[SearchHit]:
        candidates: dict[str, tuple[SearchHit, float]] = {}
        for results in (
            self.vector_search.search(query, top_k=top_k, filters=filters),
            self.keyword_search.search(query, top_k=top_k, filters=filters),
        ):
            for rank, hit in enumerate(results, start=1):
                previous = candidates.get(hit.chunk.id, (hit, 0.0))[1]
                candidates[hit.chunk.id] = (hit, previous + 1.0 / (rank_constant + rank))
        ranked = [SearchHit(hit.chunk, score) for hit, score in candidates.values()]
        return sorted(ranked, key=lambda hit: (-hit.score, hit.chunk.id))[:top_k]