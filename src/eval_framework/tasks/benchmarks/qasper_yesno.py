from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.tasks.base import NO_SUBJECT, BaseTask, Language, ResponseType


class QASPERYesNo(BaseTask[str]):
    NAME = "QASPERYesNo"
    DATASET_PATH = "allenai/qasper-yesno"
    SAMPLE_SPLIT = "train"
    FEWSHOT_SPLIT = "validation"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
    SUBJECTS = [NO_SUBJECT]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question"]
    LANGUAGE = Language.ENG

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        evidence = item.get("evidence") or []
        src = " ".join(evidence) if isinstance(evidence, list) else str(evidence)
        question = item.get("question", "")
        return f"{src.strip()}\nQuestion: {question}\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        ans = item.get("answer", "Yes")
        return f" {ans}"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [" Yes", " No"]


class QASPERYesNoMC(QASPERYesNo):
    NAME = "QASPERYesNoMC"

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        evidence = item.get("evidence") or []
        src = " ".join(evidence) if isinstance(evidence, list) else str(evidence)
        question = item.get("question", "")
        return f"{src.strip()}\nQuestion: {question}\nA. Yes\nB. No\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        ans = item.get("answer", "Yes")
        return " A" if ans == "Yes" else " B"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [" A", " B"]
