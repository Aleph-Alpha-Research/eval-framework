from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.metrics.loglikelihood.confidence_weighted_accuracy import ConfidenceWeightedAccuracy
from eval_framework.metrics.loglikelihood.dcs import DistributionalCorrectnessScore
from eval_framework.metrics.loglikelihood.ternary import TernaryScore
from eval_framework.tasks.benchmarks.hellaswag import HELLASWAG


class GOLDENSWAG(HELLASWAG):
    """GoldenSwag dataset: https://huggingface.co/datasets/PleIAs/GoldenSwag
    available data set sections: validation"""

    NAME = "GoldenSwag"
    DATASET_PATH = "PleIAs/GoldenSwag"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "validation"


class GOLDENSWAG_IDK(GOLDENSWAG):
    NAME = "GoldenSwag_IDK"
    METRICS = [
        AccuracyLoglikelihood,
        AccuracyNormLoglikelihood,
        ConfidenceWeightedAccuracy,
        DistributionalCorrectnessScore,
        TernaryScore,
    ]

    def _get_initial_prompt_text(self, item: dict[str, Any]) -> str:
        return (
            "Complete the sentence only if you are confident, since mistakes may be penalised, while correct "
            "completions receive points. It is acceptable to answer with 'I do not know' if you are unsure, "
            "and you will receive 0 points."
        )

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        completions = super()._get_possible_completions(item)
        return (completions or []) + [" I do not know."]
