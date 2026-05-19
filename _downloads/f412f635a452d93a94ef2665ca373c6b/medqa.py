"""
MedQA (English): Open-domain medical question answering from medical exams.
"""

from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.metrics.loglikelihood.bits_per_byte import BitsPerByteLoglikelihood
from eval_framework.tasks.base import NO_SUBJECT, BaseTask, Language, ResponseType
from eval_framework.tasks.utils import get_n_letters


class MedQACloze(BaseTask[str]):
    """MedQA cloze (loglikelihood over choice text)."""

    NAME = "MedQACloze"
    DATASET_PATH = "davidheineman/medqa-en"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "dev"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood, BitsPerByteLoglikelihood]
    SUBJECTS = [NO_SUBJECT]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question"]
    LANGUAGE = Language.ENG

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"Question: {item['question']}\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        choices = item.get("choices", [])
        answer_idx = item.get("answer_idx")
        if answer_idx is None or not choices:
            return None
        return f" {choices[int(answer_idx)]}"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str]:
        choices = item.get("choices", [])
        return [f" {c}" for c in choices]

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        ground_truth = self._get_ground_truth(item)
        assert ground_truth is not None
        return f"{self._get_cue_text(item)}{ground_truth}"


class MedQAMC(MedQACloze):
    """MedQA multiple choice (loglikelihood over A/B/C/D/...)."""

    NAME = "MedQAMC"

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        self.keys = get_n_letters(5)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        question = item["question"]
        choices = item.get("choices", [])
        options = "\n".join(f"{label}. {choice}" for label, choice in zip(self.keys, choices))
        return f"Question: {question}\n{options}\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        answer_idx = item.get("answer_idx")
        if answer_idx is None:
            return None
        return f" {self.keys[int(answer_idx)]}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str]:
        return [f" {label}" for label in self.keys]


class MedQAMC_OLMES(MedQAMC):
    """
    MedQA multiple choice with OLMES-style prompt: space before each label (" A.", " B.", ...).
    """

    NAME = "MedQAMC_OLMES"
    FEWSHOT_SPLIT = "train"

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        question = item["question"]
        choices = item.get("choices", [])
        options = "\n".join(f" {label}. {choice}" for label, choice in zip(self.keys, choices))
        return f"Question: {question}\n{options}\n"
