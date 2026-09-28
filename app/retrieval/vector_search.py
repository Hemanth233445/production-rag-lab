"""In-memory cosine search with exact-match metadata filters."""

from dataclasses import dataclass

from app.ingestion.embeddings import Embedder, HashingEmbedder
from app.models import Chunk


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


class InMemoryVectorIndex:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or HashingEmbedder()
        self._entries: dict[str, tuple[Chunk, list[float]]] = {}

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._entries[chunk.id] = (chunk, self.embedder.embed(chunk.text))

    @staticmethod
    def _matches(chunk: Chunk, filters: dict | None) -> bool:
        return all(chunk.metadata.get(key) == value for key, value in (filters or {}).items())

    def search(self, query: str, top_k: int = 8, filters: dict | None = None) -> list[SearchHit]:
        if top_k <= 0:
            return []
        query_vector = self.embedder.embed(query)
        hits = []
        for chunk, vector in self._entries.values():
            if self._matches(chunk, filters):
                score = sum(left * right for left, right in zip(query_vector, vector))
                hits.append(SearchHit(chunk, score))
        return sorted(hits, key=lambda hit: (-hit.score, hit.chunk.id))[:top_k]