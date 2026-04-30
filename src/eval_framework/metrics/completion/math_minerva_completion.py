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

    English Minerva extraction is the default. Subclasses select other
    final-answer styles by overriding ``COT_STYLE`` / ``RELAXED``.
    """

    NAME = "Math Minerva Completion"
    KEYS = ["Exact", "Exact Flex"]
    AGGREGATORS = [PassAtK()]

    # Defaults; subclasses override these class attributes to define variants.
    COT_STYLE: str = "minerva"
    RELAXED: bool = False

    def __init__(
        self,
        use_cot: bool = True,
        cot_style: str | None = None,
        relaxed: bool | None = None,
    ) -> None:
        self.use_cot = use_cot
        self.cot_style = cot_style if cot_style is not None else self.COT_STYLE
        self.relaxed = relaxed if relaxed is not None else self.RELAXED

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
        all_candidates = extract_answers(
            raw,
            use_cot=self.use_cot,
            cot_style=self.cot_style,
            relaxed=self.relaxed,
        )

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

    NAME = "Math Minerva Completion Relaxed"
    RELAXED = True


class MathMinervaCompletionDE(MathMinervaCompletion):
    """MathMinervaCompletion with German final-answer extraction (``Finale Antwort: …``)."""

    NAME = "Math Minerva Completion DE"
    COT_STYLE = "minerva_de"


class MathMinervaCompletionRelaxedDE(MathMinervaCompletionDE):
    """MathMinervaCompletionDE with relaxed=True by default."""

    NAME = "Math Minerva Completion Relaxed DE"
    RELAXED = True
