from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.metrics.loglikelihood.confidence_weighted_accuracy import ConfidenceWeightedAccuracy
from eval_framework.metrics.loglikelihood.distributional_correctness_score import DistributionalCorrectnessScore
from eval_framework.metrics.loglikelihood.ternary import TernaryScore
from eval_framework.tasks.base import BaseTask, Language, ResponseType


class COPA(BaseTask[str]):
    """COPA dataset: https://huggingface.co/datasets/aps/super_glue"""

    NAME = "COPA"
    DATASET_PATH = "aps/super_glue"
    SAMPLE_SPLIT = "validation"  # 100 examples (same split as lm-eval)
    FEWSHOT_SPLIT = "test"  # 500 examples
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
    SUBJECTS = ["copa"]
    PERTURBATION_UNMODIFIABLE_WORDS = ["because", "therefore"]
    LANGUAGE = Language.ENG

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        connector = {
            "cause": "because",
            "effect": "therefore",
        }[item["question"]]
        return item["premise"].strip()[:-1] + f" {connector} "

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        correct_choice = item["choice1"] if item["label"] == 0 else item["choice2"]
        return f"{self.convert_choice(correct_choice)}"

    def convert_choice(self, choice: str) -> str:
        return choice[0].lower() + choice[1:]

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        choices = [self.convert_choice(item["choice1"]), self.convert_choice(item["choice2"])]
        return choices

class COPA_IDK(COPA):
    NAME = "COPA_IDK"
    METRICS = [
        AccuracyLoglikelihood,
        AccuracyNormLoglikelihood,
        ConfidenceWeightedAccuracy,
        DistributionalCorrectnessScore,
        TernaryScore
    ]

    def _get_initial_prompt_text(self, item: dict[str, Any]) -> str:
        return (
            "Complete the sentence only if you are confident, since mistakes may be penalised, while correct answers receive points. "
            "It is acceptable to answer with 'I do not know' if you are unsure, and you will receive 0 points."
        )

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        completions = super()._get_possible_completions(item)
        return (completions or []) + ["I do not know."]
