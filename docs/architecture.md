# Architecture

The request path is intentionally split into retrieval, generation, and grounding:

1. Ingestion turns plain text, Markdown, or PDF files into stable, metadata-bearing chunks.
2. Hybrid retrieval combines local cosine feature-hash search and BM25-style keyword search with reciprocal-rank fusion. Exact metadata filters are applied before scoring.
3. A lexical reranker narrows the candidate list. Production deployments should replace this baseline with an evaluated cross-encoder or hosted reranker.
4. Generation uses an OpenAI-compatible endpoint when `OPENAI_API_KEY` is configured; otherwise it returns a cited extractive answer.
5. The grounding guard rejects responses with missing or out-of-context citation IDs.

The in-memory index and feature-hashing embedder keep the reference runnable without infrastructure, but are not substitutes for durable storage, semantic embeddings, access-control enforcement, or monitoring. Persist and scope metadata filters at the storage boundary before exposing this design to multiple tenants.

Retrieval evaluation uses `evaluation/golden_dataset.json` and reports source-level Recall@k. `evaluation/ragas_eval.py` is an optional adapter for Ragas datasets; install Ragas separately when running it.