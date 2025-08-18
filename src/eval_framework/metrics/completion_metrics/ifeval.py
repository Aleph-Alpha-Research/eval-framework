from typing import Any

from eval_framework.external.ifeval_impl.utils import process_results
from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import BaseMetricContext, Completion, extract_context_metric


class IFEvalMetricContext(BaseMetricContext):
    key: int
    instruction_id_list: list[str]
    prompt: str
    additional_kwargs: list[dict[str, Any]]


class IFEvalMetric(BaseMetric[Completion]):
    NAME = "IFEval"

    def calculate(self, response: Completion) -> list[MetricResult]:
        context = extract_context_metric(response, IFEvalMetricContext)

        if response.error is not None:
            return [
                MetricResult(
                    metric_name=f"{self.NAME}/prompt_level_strict_acc",
                    value=None,
                    higher_is_better=True,
                    error=response.error,
                ),
                MetricResult(
                    metric_name=f"{self.NAME}/prompt_level_loose_acc",
                    value=None,
                    higher_is_better=True,
                    error=response.error,
                ),
            ]

        grading = process_results(context, [response.completion])

        results = [
            MetricResult(
                metric_name=f"{self.NAME}/prompt_level_strict_acc",
                value=float(grading["prompt_level_strict_acc"]),
                higher_is_better=True,
                error=response.error,
            ),
            MetricResult(
                metric_name=f"{self.NAME}/prompt_level_loose_acc",
                value=float(grading["prompt_level_loose_acc"]),
                higher_is_better=True,
                error=response.error,
            ),
        ]
        # this framework does not support a custom aggregation step (see agg_inst_level_acc()) so work around
        # by returning the result for each instruction as a separate MetricResult
        results += [
            MetricResult(
                metric_name=f"{self.NAME}/inst_level_strict_acc",
                value=float(v),
                higher_is_better=True,
                error=response.error,
            )
            for v in grading["inst_level_strict_acc"]
        ]
        results += [
            MetricResult(
                metric_name=f"{self.NAME}/inst_level_loose_acc",
                value=float(v),
                higher_is_better=True,
                error=response.error,
            )
            for v in grading["inst_level_loose_acc"]
        ]
        return results
