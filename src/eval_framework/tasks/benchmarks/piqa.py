from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.metrics.loglikelihood.bits_per_byte import BitsPerByteLoglikelihood
from eval_framework.metrics.loglikelihood.confidence_weighted_accuracy import ConfidenceWeightedAccuracy
from eval_framework.metrics.loglikelihood.dcs import DistributionalCorrectnessScore
from eval_framework.metrics.loglikelihood.ternary import TernaryScore
from eval_framework.tasks.base import NO_SUBJECT, BaseTask, Language, ResponseType
from eval_framework.tasks.utils import get_n_letters


class PIQA(BaseTask[str]):
    """PIQA dataset: https://huggingface.co/datasets/ybisk/piqa"""

    NAME = "PIQA"
    DATASET_PATH = "ybisk/piqa"
    HF_REVISION = "6b3aceb3276e5ab7e51895d73151a718690af38c"
    SAMPLE_SPLIT = "validation"  # 1838 examples (same split as lm-eval)
    FEWSHOT_SPLIT = "test"  # 3084 examples
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood, BitsPerByteLoglikelihood]
    SUBJECTS = [NO_SUBJECT]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question"]
    LANGUAGE = Language.ENG

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"Question: {item['goal']}\n"

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        ground_truth = self._get_ground_truth(item)
        assert ground_truth is not None
        return f"{self._get_cue_text(item)}{ground_truth}"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        truth = item["sol1"] if item["label"] == 0 else item["sol2"]
        return f" {truth}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [f" {choice}" for choice in [item["sol1"], item["sol2"]]]


class PIQA_OLMES(PIQA):
    """
    PIQA with OLMES-style prompt: options shown with space-prefixed labels (" A.", " B.");
    loglikelihood over " A"/" B".
    """

    NAME = "PIQA_OLMES"
    FEWSHOT_SPLIT = "train"

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        self.keys = get_n_letters(2)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        goal = item["goal"]
        choices = [item["sol1"], item["sol2"]]
        options = "\n".join(f" {key}. {choice}" for key, choice in zip(self.keys, choices))
        return f"Goal: {goal}\n{options}\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        idx = 0 if item["label"] == 0 else 1
        return f" {self.keys[idx]}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [f" {key}" for key in self.keys]


class PIQA_IDK(PIQA):
    NAME = "PIQA_IDK"
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
            "answers receive points. It is acceptable to answer with 'I do not know' if you are unsure, and "
            "you will receive 0 points."
        )

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        completions = super()._get_possible_completions(item)
        return (completions or []) + [" I do not know"]
