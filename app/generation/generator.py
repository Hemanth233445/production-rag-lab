"""Grounded extractive fallback and optional OpenAI-compatible generation."""

import json
import re
from urllib.request import Request, urlopen

from app.config import Settings
from app.retrieval.vector_search import SearchHit


class ExtractiveGenerator:
    def generate(self, question: str, hits: list[SearchHit]) -> str:
        if not hits:
            return "I don't have enough information in the indexed sources to answer that."
        question_terms = set(re.findall(r"[a-z0-9]+", question.lower()))
        best: tuple[int, str, str] | None = None
        for hit in hits:
            sentences = re.split(r"(?<=[.!?])\s+", hit.chunk.text)
            for sentence in sentences:
                terms = set(re.findall(r"[a-z0-9]+", sentence.lower()))
                candidate = (len(question_terms & terms), sentence.strip(), hit.chunk.id)
                if candidate[1] and (best is None or candidate[0] > best[0]):
                    best = candidate
        if best is None or best[0] == 0:
            return "I don't have enough information in the indexed sources to answer that."
        return f"From the available sources: {best[1]} [{best[2]}]"


class OpenAICompatibleGenerator:
    def __init__(self, settings: Settings) -> None:
        if not settings.api_key:
            raise ValueError("OPENAI_API_KEY is required for the hosted generator")
        self.settings = settings

    def generate(self, question: str, hits: list[SearchHit]) -> str:
        from app.generation.prompt import build_prompt

        body = json.dumps({
            "model": self.settings.model,
            "messages": [{"role": "user", "content": build_prompt(question, hits)}],
            "temperature": 0,
        }).encode("utf-8")
        request = Request(
            f"{self.settings.api_base_url}/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {self.settings.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read())
        return payload["choices"][0]["message"]["content"]