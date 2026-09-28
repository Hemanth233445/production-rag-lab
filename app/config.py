"""Environment-backed application settings."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    generation_mode: str = "local"
    api_key: str | None = None
    model: str = "gpt-4o-mini"
    api_base_url: str = "https://api.openai.com/v1"
    chunk_size: int = 180
    chunk_overlap: int = 30
    retrieval_top_k: int = 8
    rerank_top_k: int = 4

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            generation_mode=os.getenv("GENERATION_MODE", "local").lower(),
            api_key=os.getenv("OPENAI_API_KEY") or None,
            model=os.getenv("GENERATION_MODEL", "gpt-4o-mini"),
            api_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "180")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "30")),
            retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "8")),
            rerank_top_k=int(os.getenv("RERANK_TOP_K", "4")),
        )