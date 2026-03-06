from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.metrics.loglikelihood.bits_per_byte import BitsPerByteLoglikelihood
from eval_framework.metrics.loglikelihood.confidence_weighted_accuracy import ConfidenceWeightedAccuracy
from eval_framework.metrics.loglikelihood.dcs import DistributionalCorrectnessScore
from eval_framework.metrics.loglikelihood.ternary import TernaryScore
from eval_framework.tasks.base import BaseTask, Language, ResponseType
from eval_framework.tasks.utils import get_n_letters

ANSWER_STR_TO_NUM = {"1": 0, "2": 1}


class WINOGRANDE(BaseTask[str]):
    """WINOGRANDE dataset: https://huggingface.co/datasets/allenai/winogrande"""

    NAME = "Winogrande"
    DATASET_PATH = "allenai/winogrande"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood, BitsPerByteLoglikelihood]
    SUBJECTS = ["winogrande_xl"]
    PERTURBATION_UNMODIFIABLE_WORDS = ["1", "2"]
    LANGUAGE = Language.ENG

    def _extract_question(self, item: dict) -> str:
        question, _ = item["sentence"].split("_")
        question = question.replace("  ", " ")
        return question.strip()

    def _extract_choices(self, item: dict) -> list[str]:
        _, choice_suffix = item["sentence"].split("_")
        choice_suffix = choice_suffix.replace("  ", " ")
        choices = [choice + choice_suffix for choice in [item["option1"], item["option2"]]]
        return choices

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"{self._extract_question(item)}"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        choices = self._extract_choices(item)
        return f" {choices[ANSWER_STR_TO_NUM[item['answer']]]}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [f" {choice}" for choice in self._extract_choices(item)]


class WINOGRANDE_OLMES(WINOGRANDE):
    """
    Winogrande with OLMES-style prompt: options shown with space-prefixed labels (" A.", " B.");
    loglikelihood over " A"/" B".
    """

    NAME = "Winogrande_OLMES"

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        self.keys = get_n_letters(2)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        question = self._extract_question(item)
        choices = self._extract_choices(item)
        options = "\n".join(f" {key}. {choice}" for key, choice in zip(self.keys, choices))
        return f"{question}\n{options}\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        idx = ANSWER_STR_TO_NUM[item["answer"]]
        return f" {self.keys[idx]}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [f" {key}" for key in self.keys]


class WINOGRANDE_IDK(WINOGRANDE):
    NAME = "Winogrande_IDK"
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
        return (completions or []) + [" I do not know."]
