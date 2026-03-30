"""DROP completion metrics: F1 and exact match."""

from eval_framework.external.drop_process_results import process_results
from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import (
    BaseMetricContext,
    Completion,
    extract_context_metric,
)


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
                MetricResult(
                    metric_name=name,
                    value=None,
                    higher_is_better=True,
                    error=response.error,
                )
                for name in [n.strip() for n in self.NAME.split("/")]
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
        out = process_results(doc, pred_spans)

        return [
            MetricResult(
                metric_name=name,
                value=out[key],
                higher_is_better=True,
                error=response.error,
            )
            for name, key in zip([n.strip() for n in self.NAME.split("/")], self.KEYS)
        ]
