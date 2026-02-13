from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.metrics.loglikelihood.confidence_weighted_accuracy import ConfidenceWeightedAccuracy
from eval_framework.metrics.loglikelihood.dcs import DistributionalCorrectnessScore
from eval_framework.metrics.loglikelihood.ternary import TernaryScore
from eval_framework.tasks.base import BaseTask, Language, ResponseType


class WINOGENDER(BaseTask[str]):
    """WINOGENDER dataset: https://huggingface.co/datasets/datasets/oskarvanderwal/winogender"""

    NAME = "Winogender"
    DATASET_PATH = "oskarvanderwal/winogender"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "test"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
    SUBJECTS = ["all"]
    LANGUAGE = Language.ENG

    def _extract_question(self, item: dict) -> str:
        """Format question according to Llama paper."""
        return f"{item['sentence']} '{item['pronoun'].capitalize()}' refers to"

    def _extract_choices(self, item: dict) -> list[str]:
        choices = item["occupation"], item["participant"]
        # add "the" to any choice that isn't "someone" (else it's ungrammatical)
        return [f"the {c}" if c.lower() != "someone" else c for c in choices]

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return self._extract_question(item)

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        choices = self._extract_choices(item)
        return f" {choices[item['label']]}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [f" {choice}" for choice in self._extract_choices(item)]


class WINOGENDER_IDK(WINOGENDER):
    NAME = "Winogender_IDK"
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
            "completions receive points. It is acceptable to answer with 'don't know' if you are unsure, and "
            "you will receive 0 points."
        )

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        completions = super()._get_possible_completions(item)
        return (completions or []) + [" don't know"]
