"""FastAPI entry point for the local RAG reference implementation."""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.config import Settings
from app.generation.generator import ExtractiveGenerator, OpenAICompatibleGenerator
from app.guardrails.grounding import validate_grounding
from app.ingestion.chunker import chunk_document
from app.models import Document
from app.retrieval.hybrid_search import HybridSearch
from app.retrieval.keyword_search import KeywordSearch
from app.retrieval.reranker import rerank
from app.retrieval.vector_search import InMemoryVectorIndex


class IngestRequest(BaseModel):
    text: str = Field(min_length=1)
    source: str = "api"
    metadata: dict = Field(default_factory=dict)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    filters: dict = Field(default_factory=dict)


class RAGEngine:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.vector_index = InMemoryVectorIndex()
        self.keyword_search = KeywordSearch()
        self.hybrid_search = HybridSearch(self.vector_index, self.keyword_search)
        if self.settings.generation_mode == "local":
            self.generator = ExtractiveGenerator()
        elif self.settings.generation_mode == "hosted":
            self.generator = OpenAICompatibleGenerator(self.settings)
        else:
            raise ValueError("GENERATION_MODE must be 'local' or 'hosted'")

        chandra = self.settings.chandra_path
        if chandra and chandra.exists():
            from app.ingestion.chandra_ingestor import ChandraIngestor

            ingestor = ChandraIngestor(chandra)
            for document in ingestor.iter_documents():
                self.ingest(document.text, document.metadata.get("source", "chandra"), document.metadata)

    def ingest(self, text: str, source: str, metadata: dict | None = None) -> int:
        document_metadata = {"source": source, **(metadata or {})}
        chunks = chunk_document(
            Document(text=text, metadata=document_metadata),
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        self.vector_index.add(chunks)
        self.keyword_search.add(chunks)
        return len(chunks)

    def query(self, question: str, filters: dict | None = None) -> dict:
        candidates = self.hybrid_search.search(
            question, top_k=self.settings.retrieval_top_k, filters=filters
        )
        hits = rerank(question, candidates, top_k=self.settings.rerank_top_k)
        try:
            draft = self.generator.generate(question, hits)
        except Exception:
            draft = ExtractiveGenerator().generate(question, hits)
        answer = validate_grounding(draft, hits)
        return {
            "answer": answer,
            "citations": [
                {"id": hit.chunk.id, "source": hit.chunk.metadata.get("source"), "text": hit.chunk.text}
                for hit in hits
                if f"[{hit.chunk.id}]" in answer
            ],
        }


def create_app(engine: RAGEngine | None = None) -> FastAPI:
    rag = engine or RAGEngine()
    application = FastAPI(title="Production RAG Lab", version="0.1.0")

    @application.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @application.post("/ingest")
    def ingest(request: IngestRequest) -> dict:
        return {"chunks_indexed": rag.ingest(request.text, request.source, request.metadata)}

    @application.post("/query")
    def query(request: QueryRequest) -> dict:
        return rag.query(request.question, request.filters)

    return application


app = create_app()