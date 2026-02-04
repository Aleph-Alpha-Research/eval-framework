"""
MedQA (English): Open-domain medical question answering from medical exams.
"""

from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.tasks.base import NO_SUBJECT, BaseTask, Language, ResponseType
from eval_framework.tasks.utils import get_n_letters


class MedQACloze(BaseTask[str]):
    """MedQA cloze (loglikelihood over choice text)."""

    NAME = "MedQACloze"
    DATASET_PATH = "davidheineman/medqa-en"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "dev"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
    SUBJECTS = [NO_SUBJECT]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question"]
    LANGUAGE = Language.ENG

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"Question: {item['question']}\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str:
        choices = item.get("choices", [])
        answer_idx = int(item.get("answer_idx", 0))
        return f" {choices[answer_idx]}"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str]:
        choices = item.get("choices", [])
        return [f" {c}" for c in choices]


class MedQAMC(MedQACloze):
    """MedQA multiple choice (loglikelihood over A/B/C/D/...)."""

    NAME = "MedQAMC"

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        question = item["question"]
        choices = item.get("choices", [])
        labels = get_n_letters(5)
        options = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices))
        return f"Question: {question}\n{options}\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str:
        labels = get_n_letters(5)
        label = labels[int(item.get("answer_idx", 0))]
        return f" {label}"
    
    def _get_possible_completions(self, item: dict[str, Any]) -> list[str]:
        labels = get_n_letters(5)
        return [f" {label}" for label in labels]
