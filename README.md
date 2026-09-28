# Production RAG Lab

A local-first Retrieval-Augmented Generation reference implementation demonstrating:

`Hybrid Retrieval` · `Metadata Filtering` · `Reranking` · `Grounded Generation` · `Citation Generation` · `RAG Evaluation` · `Fallback Handling`

This is a modular engineering lab, not a generic “chat with PDF” application. It exposes a small API and keeps retrieval, generation, grounding, and evaluation as replaceable components. The full default path runs locally without credentials, hosted services, or persistent infrastructure. Its production-oriented design is a reference; the included implementations use local baselines.

## Run locally

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

The service listens at `http://127.0.0.1:8000`. Ingest text and metadata with `POST /ingest`, then ask a question with `POST /query`. The in-memory index resets when the process restarts.

```json
{"text":"The service falls back to extractive answers when the API is unavailable.","source":"operations.md","metadata":{"team":"platform"}}
```

```json
{"question":"What happens when the API is unavailable?","filters":{"team":"platform"}}
```

The application uses cited extractive generation locally by default. To opt into an OpenAI-compatible hosted generator, set `GENERATION_MODE=hosted` and provide `OPENAI_API_KEY`. PDF loading is available through `app.ingestion.loader` and requires `pypdf`, included in the requirements.

## Evaluate

```powershell
python -m pytest
```

Retrieval evaluation can be run from Python using `evaluation.retrieval_eval.load_golden_dataset` and `evaluate_retrieval`. It reports source-level Recall@k. Ragas remains optional; install its evaluation dependencies with `python -m pip install -r requirements-eval.txt`, then pass a Ragas-compatible dataset to `evaluation.ragas_eval.run_ragas_evaluation`.

## Production boundaries

The included in-memory vector index, feature-hashing embeddings, BM25-style search, and lexical reranker are transparent local baselines. Replace them with durable vector/keyword stores, semantic embeddings, and an evaluated reranker for production. Enforce tenant authorization in the data layer; query metadata filters are not an access-control system. See [docs/architecture.md](docs/architecture.md) for the request flow and boundaries.