from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.tasks.base import NO_SUBJECT, BaseTask, Language, ResponseType


SIMPLETOM_SUBJECTS = [
    "behavior-qa",
    "judgment-qa",
    "mental-state-qa",
]


class SimpleToMMC(BaseTask[str]):
    NAME = "SimpleToMMC"
    DATASET_PATH = "allenai/SimpleToM"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "test"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
    SUBJECTS = SIMPLETOM_SUBJECTS
    PERTURBATION_UNMODIFIABLE_WORDS = ["Story", "Question"]
    LANGUAGE = Language.ENG

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        story = item.get("story", "")
        question = item.get("question", "")
        choices = item.get("choices", {}).get("text", [])
        options = "\n".join(f"{l}. {t}" for l, t in zip(["A", "B"], choices))
        return f"Story:\n{story}\n\nQuestion: {question}\n{options}\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        labels = ["A", "B"]
        answer_key = item.get("answerKey", "")
        gt_idx = labels.index(answer_key)
        return f" {labels[gt_idx]}"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [" A", " B"]

class SimpleToMCloze(SimpleToMMC):
    NAME = "SimpleToMCloze"

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        story = item.get("story", "")
        question = item.get("question", "")
        return f"Story:\n{story}\n\nQuestion: {question}\n"
    
    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        labels = ["A", "B"]
        answer_key = item.get("answerKey", "")
        gt_idx = labels.index(answer_key)
        texts = item.get("choices", {}).get("text", [])
        return f" {texts[gt_idx]}"
    
    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [f" {t}" for t in item.get("choices", {}).get("text", [])]
