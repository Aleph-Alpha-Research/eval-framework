"""
SciRIFF Yes/No: Subset of yes/no questions from the SciRIFF dataset.
"""

from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.tasks.base import NO_SUBJECT, BaseTask, Language, ResponseType


class SciRIFFYesNo(BaseTask[str]):
    """SciRIFF yes/no cloze (loglikelihood over Yes/No)."""

    NAME = "SciRIFFYesNo"
    DATASET_PATH = "allenai/sciriff-yesno"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "validation"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
    SUBJECTS = [NO_SUBJECT]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question", "Context"]
    LANGUAGE = Language.ENG

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        context = item.get("context", "").strip()
        question = item.get("question", "")
        return f"Context: {context}\nQuestion: {question}\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        ans = item.get("answer", "Yes")
        return f" {ans}"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [" Yes", " No"]


class SciRIFFYesNoMC(SciRIFFYesNo):
    """SciRIFF yes/no multiple choice (loglikelihood over A/B)."""

    NAME = "SciRIFFYesNoMC"

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        context = item.get("context", "").strip()
        question = item.get("question", "")
        return f"Context: {context}\nQuestion: {question}\nA. Yes\nB. No\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        ans = item.get("answer", "Yes")
        return " A" if ans == "Yes" else " B"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [" A", " B"]
