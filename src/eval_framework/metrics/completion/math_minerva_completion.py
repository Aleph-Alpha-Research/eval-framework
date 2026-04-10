"""
Minerva-style MATH completion metric: exact_match and exact_match_flex.
"""

from eval_framework.metrics.aggregators.aggregators import PassAtK
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
    KEYS = ["Exact", "Exact Flex"]
    AGGREGATORS = [PassAtK()]

    def __init__(
        self,
        use_cot: bool = True,
        cot_style: str = "minerva",
        relaxed: bool = False,
    ) -> None:
        self.use_cot = use_cot
        self.cot_style = cot_style
        self.relaxed = relaxed

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error:
            return [
                MetricResult(
                    metric_name=x,
                    value=None,
                    higher_is_better=True,
                    error=response.error,
                )
                for x in self.NAMES
            ]

        gold = response.ground_truth
        if isinstance(gold, list):
            gold = gold[0] if gold else None
        if not gold:
            return [
                MetricResult(
                    metric_name=x,
                    value=None,
                    higher_is_better=True,
                    error="No ground truth available",
                )
                for x in self.NAMES
            ]

        raw = response.raw_completion or response.completion
        all_candidates = extract_answers(raw, use_cot=self.use_cot, cot_style=self.cot_style, relaxed=self.relaxed)

        exact_match = 0.0
        if all_candidates:
            primary = all_candidates[0]
            if is_equiv_minerva(primary, gold):
                exact_match = 1.0

        exact_match_flex = float(
            any(
                is_equiv_minerva(candidate, gold) or is_equiv_hendrycks(candidate, gold) for candidate in all_candidates
            )
        )

        return [
            MetricResult(metric_name=name, value=value, higher_is_better=True)
            for name, value in zip(self.NAMES, [exact_match, exact_match_flex])
        ]


class MathMinervaCompletionRelaxed(MathMinervaCompletion):
    """MathMinervaCompletion with relaxed=True by default (flexible final-answer matching)."""

    def __init__(
        self,
        use_cot: bool = True,
        cot_style: str = "minerva",
        relaxed: bool = True,
    ) -> None:
        super().__init__(use_cot=use_cot, cot_style=cot_style, relaxed=relaxed)
