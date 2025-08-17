from collections import Counter
from typing import Any

from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Completion


class F1(BaseMetric[Completion]):
    NAME = "F1"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        ground_truths = [gt for gt in response.ground_truth_list if gt is not None]
        if not ground_truths:
            return [MetricResult(metric_name=self.NAME, value=0.0, higher_is_better=True, error=response.error)]

        hyp_tokens = response.completion.lower().split()
        f1_scores = [calculate_f1(gt.lower().split(), hyp_tokens) for gt in ground_truths]
        max_f1 = max(f1_scores)

        return [MetricResult(metric_name=self.NAME, value=max_f1, higher_is_better=True, error=response.error)]


def calculate_f1(ref_tokens: list[Any], hyp_tokens: list[Any]) -> float:
    """Calculate F1 score between two texts based on token overlap."""
    if not ref_tokens and not hyp_tokens:
        return 1.0
    if not ref_tokens or not hyp_tokens:
        return 0.0

    common = Counter(ref_tokens) & Counter(hyp_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(hyp_tokens)
    recall = num_same / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)
