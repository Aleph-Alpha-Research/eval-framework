from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.tasks.base import NO_SUBJECT, BaseTask, Language, ResponseType


class CommonsenseQACloze(BaseTask[str]):
    """CommonsenseQA dataset: https://huggingface.co/datasets/tau/commonsense_qa"""

    NAME = "CommonsenseQACloze"
    DATASET_PATH = "tau/commonsense_qa"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "validation"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
    SUBJECTS = [NO_SUBJECT]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question"]
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"Question: {item['question']}\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        # choices["label"] contains letters A-E; "answerKey" is the correct label.
        labels = item["choices"]["label"]
        texts = item["choices"]["text"]
        correct_label = item["answerKey"]
        correct_index = labels.index(correct_label)
        return f" {texts[correct_index]}"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [f" {choice}" for choice in item["choices"]["text"]]


class CommonsenseQAMC(CommonsenseQACloze):
    """Multiple-choice variant of CommonsenseQA where the model selects a letter (A-E)."""

    NAME = "CommonsenseQAMC"

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        question = item["question"]
        texts = item["choices"]["text"]
        labels = item["choices"]["label"]
        options = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, texts))
        return f"Question: {question}\n{options}\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        correct_label = item["answerKey"]
        return f" {correct_label}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        labels = item["choices"]["label"]
        return [f" {label}" for label in labels]
