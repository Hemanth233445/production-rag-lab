"""Recall@k evaluation over the repository's source-based golden dataset."""

import json
from pathlib import Path
from typing import Iterable

from app.retrieval.hybrid_search import HybridSearch


def evaluate_retrieval(
    retriever: HybridSearch,
    cases: Iterable[dict],
    top_k: int = 5,
) -> dict:
    results = []
    for case in cases:
        expected = set(case["expected_sources"])
        hits = retriever.search(case["question"], top_k=top_k, filters=case.get("filters"))
        found = {str(hit.chunk.metadata.get("source", "")) for hit in hits}
        results.append({"question": case["question"], "recall": len(expected & found) / len(expected) if expected else 1.0})
    return {"recall_at_k": sum(row["recall"] for row in results) / len(results) if results else 0.0, "cases": results}


def load_golden_dataset(path: str | Path | None = None) -> list[dict]:
    dataset_path = Path(path) if path else Path(__file__).with_name("golden_dataset.json")
    return json.loads(dataset_path.read_text(encoding="utf-8"))