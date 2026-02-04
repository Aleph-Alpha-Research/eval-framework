from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.tasks.base import NO_SUBJECT, BaseTask, Language, ResponseType
from eval_framework.metrics.completion.accuracy_completion import AccuracyCompletion


COPYCOLORS_SUBSETS = [
    "2_answer_choices",
    "3_answer_choices",
    "4_answer_choices",
    "5_answer_choices",
    "6_answer_choices",
    "7_answer_choices",
    "8_answer_choices",
    "9_answer_choices",
    "10_answer_choices",
    "11_answer_choices"
]


class CopyColorsCloze(BaseTask[str]):
    NAME = "CopyColorsCloze"
    DATASET_PATH = "sarahwie/copycolors_mcqa"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "validation"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood, AccuracyCompletion]
    SUBJECTS = COPYCOLORS_SUBSETS
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question", "Answer"]
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        question = item.get("question", "")
        return f"Question: {question}\n"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        return f" {item.get('answer')}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        texts = item.get("choices", {}).get("text", [])
        return [f" {t}" for t in texts]


class CopyColorsMC(CopyColorsCloze):
    NAME = "CopyColorsMC"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood, AccuracyCompletion]
    SUBJECTS = COPYCOLORS_SUBSETS
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question", "Answer"]

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        question = item.get("question", "")
        texts = item.get("choices", {}).get("text", [])
        labels = item.get("choices", {}).get("label", [])
        options = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
        return f"Question: {question}\n{options}\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        labels = item.get("choices", {}).get("label", [])
        return f" {labels[item.get('answerKey', 0)]}" if 0 <= item.get('answerKey', 0) < len(labels) else None

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        labels = item.get("choices", {}).get("label", [])
        return [f" {l}" for l in labels]
