"""
Minerva-style MATH completion metric: exact_match and exact_match_flex.
"""

from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.metrics.completion.minerva_math_utils import (
    extract_answers,
    is_equiv_hendrycks,
    is_equiv_minerva,
)
from eval_framework.shared.types import Completion


class MathMinervaCompletion(BaseMetric[Completion]):
    """
    Minerva MATH: reports Exact Match and Exact Match (Flex).
    Uses raw_completion to extract multiple candidates; primary for exact_match,
    all candidates with both Minerva and Hendrycks equivalence for exact_match_flex.
    """

    NAME = "Math Minerva Completion"

    def __init__(
        self,
        use_cot: bool = True,
        cot_style: str = "minerva",
    ) -> None:
        self.use_cot = use_cot
        self.cot_style = cot_style

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [
                MetricResult(
                    metric_name="Exact Match",
                    value=None,
                    higher_is_better=True,
                    error=response.error,
                ),
                MetricResult(
                    metric_name="Exact Match (Flex)",
                    value=None,
                    higher_is_better=True,
                    error=response.error,
                ),
            ]

        gold = response.ground_truth
        if isinstance(gold, list):
            gold = gold[0] if gold else None
        if gold is None:
            return [
                MetricResult(metric_name="Exact Match", value=0.0, higher_is_better=True),
                MetricResult(metric_name="Exact Match (Flex)", value=0.0, higher_is_better=True),
            ]

        raw = response.raw_completion or response.completion
        all_candidates = extract_answers(raw, use_cot=self.use_cot, cot_style=self.cot_style)

        exact_match = 0.0
        if all_candidates:
            primary = all_candidates[0]
            if is_equiv_minerva(primary, gold):
                exact_match = 1.0

        exact_match_flex = 0.0
        for candidate in all_candidates:
            if exact_match_flex == 1.0:
                break
            if is_equiv_minerva(candidate, gold) or is_equiv_hendrycks(candidate, gold):
                exact_match_flex = 1.0

        return [
            MetricResult(metric_name="Exact Match", value=exact_match, higher_is_better=True),
            MetricResult(
                metric_name="Exact Match (Flex)",
                value=exact_match_flex,
                higher_is_better=True,
            ),
        ]
