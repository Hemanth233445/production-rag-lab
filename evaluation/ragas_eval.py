"""Optional Ragas adapter for externally assembled answer/context datasets."""

from importlib import import_module


def run_ragas_evaluation(dataset, metrics=None):
    try:
        ragas = import_module("ragas")
        ragas_metrics = import_module("ragas.metrics")
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install the optional `ragas` package to run this evaluation.") from exc

    selected_metrics = metrics or [
        ragas_metrics.faithfulness,
        ragas_metrics.answer_relevancy,
        ragas_metrics.context_precision,
    ]
    return ragas.evaluate(dataset=dataset, metrics=selected_metrics)