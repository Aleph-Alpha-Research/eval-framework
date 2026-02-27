"""DROP completion metrics: F1 and exact match."""

from eval_framework.external.drop_process_results import process_results
from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import BaseMetricContext, Completion, extract_context_metric


class DropMetricContext(BaseMetricContext):
    """Context for DROP completion metrics. answer_tuples: list of gold answers (each a list of strings)."""

    answer_tuples: list[list[str]]


class DropF1ExactMatch(BaseMetric[Completion]):
    """DROP F1 and exact match. Requires DropMetricContext with answer_tuples."""

    NAME = "DROP F1 / Exact Match"
    KEYS = ["f1", "exact_match"]

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [
                MetricResult(metric_name=f"{self.NAME}/f1", value=None, higher_is_better=True, error=response.error),
                MetricResult(
                    metric_name=f"{self.NAME}/exact_match", value=None, higher_is_better=True, error=response.error
                ),
            ]

        context = extract_context_metric(response, DropMetricContext)
        # Gold: list of tuples (stored as list of lists)
        answer_tuples = [list(a) for a in context.answer_tuples]
        # Parse completion: comma-separated spans or single string
        raw = (response.completion or "").strip()
        pred_spans = [s.strip() for s in raw.split(",") if s.strip()] if raw else []
        if not pred_spans:
            pred_spans = [raw]

        doc = {"answers": answer_tuples}
        results = [pred_spans]
        out = process_results(doc, results)

        return [
            MetricResult(metric_name="DROP F1", value=out["f1"], higher_is_better=True, error=response.error),
            MetricResult(
                metric_name="Exact Match", value=out["exact_match"], higher_is_better=True, error=response.error
            ),
        ]
