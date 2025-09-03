from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.tasks.base import BaseTask, Language, ResponseType

ANSWER_STR_TO_NUM = {"1": 0, "2": 1}


class WINOGRANDE(BaseTask[str]):
    """WINOGRANDE dataset: https://huggingface.co/datasets/winogrande"""

    NAME = "Winogrande"
    DATASET_PATH = "winogrande"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
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
