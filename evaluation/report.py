"""Render retrieval evaluation results as a compact Markdown report."""


def render_report(results: dict) -> str:
    lines = ["# Retrieval Evaluation", "", f"Recall@k: **{results['recall_at_k']:.3f}**", "", "| Question | Recall |", "| --- | ---: |"]
    lines.extend(f"| {case['question'].replace('|', '\\|')} | {case['recall']:.3f} |" for case in results["cases"])
    return "\n".join(lines)