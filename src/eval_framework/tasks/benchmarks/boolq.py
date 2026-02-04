from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.tasks.base import BaseTask, Language, ResponseType


class BoolQCloze(BaseTask[str]):
    """BoolQ dataset: https://huggingface.co/datasets/super_glue/viewer/boolq."""

    NAME = "BoolQCloze"
    DATASET_PATH = "aps/super_glue"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "test"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
    SUBJECTS = "boolq"
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question", "Answer", "yes", "no"]
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        passage = item["passage"]
        question = item["question"]
        return f"{passage}\nQuestion: {question}?\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        # In the HF dataset, label 1 corresponds to True/Yes.
        answer = "yes" if item["label"] == 1 else "no"
        return f" {answer}"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [" yes", " no"]


class BoolQMC(BoolQCloze):
    """
    Multiple-choice variant of BoolQ where the model selects A or B, corresponding to yes/no.
    """

    NAME = "BoolQMC"

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        passage = item["passage"]
        question = item["question"]
        options = "A. yes\nB. no"
        return f"{passage}\nQuestion: {question}?\n{options}\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        # We invert as in OLMES (gold = 1 - label) so that A corresponds to "yes".
        idx = 1 - item["label"]
        choice_labels = ["A", "B"]
        return f" {choice_labels[idx]}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [" A", " B"]

