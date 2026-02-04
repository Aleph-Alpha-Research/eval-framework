"""
MedMCQA: Multi-subject multiple-choice medical question answering.
"""

from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.tasks.base import NO_SUBJECT, BaseTask, Language, ResponseType
from eval_framework.tasks.utils import get_n_letters


class MedMCQACloze(BaseTask[str]):
    """MedMCQA cloze (loglikelihood over option text)."""

    NAME = "MedMCQACloze"
    DATASET_PATH = "openlifescienceai/medmcqa"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "validation"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
    SUBJECTS = [NO_SUBJECT]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question"]
    LANGUAGE = Language.ENG

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"Question: {item['question']}\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str:
        # cop is 1-indexed (1, 2, 3, 4) in the dataset
        cop = int(item["cop"])
        idx = cop - 1
        choices = [item["opa"], item["opb"], item["opc"], item["opd"]]
        return f" {choices[idx]}"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str]:
        choices = [item["opa"], item["opb"], item["opc"], item["opd"]]
        return [f" {c}" for c in choices]


class MedMCQAMC(MedMCQACloze):
    """MedMCQA multiple choice (loglikelihood over A/B/C/D)."""

    NAME = "MedMCQAMC"

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        question = item["question"]
        choices = [item["opa"], item["opb"], item["opc"], item["opd"]]
        labels = ["A", "B", "C", "D"]
        options = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices))
        return f"Question: {question}\n{options}\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str:
        idx = int(item["cop"]) - 1
        return f" {[" A", " B", " C", " D"][idx]}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str]e:
        labels = ["A", "B", "C", "D"]
        return [f" {l}" for l in labels]
