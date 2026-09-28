from app.generation.generator import ExtractiveGenerator
from app.guardrails.grounding import validate_grounding
from app.ingestion.chunker import chunk_document
from app.config import Settings
from app.main import RAGEngine
from app.models import Document
from app.retrieval.hybrid_search import HybridSearch
from app.retrieval.keyword_search import KeywordSearch
from app.retrieval.reranker import rerank
from app.retrieval.vector_search import InMemoryVectorIndex


def make_retriever():
    chunks = chunk_document(
        Document(
            "The platform uses a fallback when the generation API is unavailable. "
            "Every answer should cite its source document.",
            {"source": "operations.md", "team": "platform"},
        ),
        chunk_size=20,
        overlap=3,
    )
    vector = InMemoryVectorIndex()
    keyword = KeywordSearch()
    vector.add(chunks)
    keyword.add(chunks)
    return HybridSearch(vector, keyword)


def test_hybrid_search_applies_metadata_filters_and_reranks():
    retriever = make_retriever()
    hits = retriever.search("generation API fallback", filters={"team": "platform"})
    assert hits
    assert all(hit.chunk.metadata["team"] == "platform" for hit in hits)
    assert rerank("generation API fallback", hits, top_k=1)[0].chunk.id == hits[0].chunk.id
    assert retriever.search("generation API", filters={"team": "sales"}) == []


def test_extractive_generator_cites_retrieved_chunk_and_grounding_accepts_it():
    hits = make_retriever().search("generation API fallback")
    answer = ExtractiveGenerator().generate("generation API fallback", hits)
    assert validate_grounding(answer, hits) == answer
    assert f"[{hits[0].chunk.id}]" in answer or any(f"[{hit.chunk.id}]" in answer for hit in hits)


def test_grounding_rejects_unknown_or_missing_citations():
    hits = make_retriever().search("generation API fallback")
    assert "don't have enough information" in validate_grounding("Unsupported claim [unknown]", hits)
    assert "don't have enough information" in validate_grounding("Unsupported claim", hits)


def test_empty_corpus_abstains():
    assert "don't have enough information" in ExtractiveGenerator().generate("question", [])


def test_local_mode_stays_local_even_when_api_key_is_present():
    engine = RAGEngine(Settings(api_key="configured-but-not-used"))
    assert isinstance(engine.generator, ExtractiveGenerator)