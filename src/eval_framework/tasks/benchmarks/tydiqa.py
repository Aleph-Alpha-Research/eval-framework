from typing import Any

from eval_framework.metrics.completion.accuracy_completion import AccuracyCompletion
from eval_framework.tasks.base import BaseTask, Language, ResponseType

TYDIQA_LANGUAGES = [
    "arabic",
    "bengali",
    "english",
    "finnish",
    "indonesian",
    "japanese",
    "korean",
    "russian",
    "swahili",
    "telugu",
    "thai",
]

# HF secondary_task columns: id, title, context, question, answers (text, answer_start).
# Language is given by the "language" column only for the `primary_task`, so in the 
# `secondary_task` we use the id prefix (e.g. "arabic-...").


class TyDiQASecondaryTask(BaseTask[str]):
    """
    TyDiQA secondary_task only: https://huggingface.co/datasets/google-research-datasets/tydiqa

    Uses the secondary_task subset and exposes the dataset's language column as SUBJECTS
    (one subject per TYDIQA_LANGUAGES entry).
    """

    NAME = "TyDiQASecondaryTask"
    DATASET_PATH = "google-research-datasets/tydiqa"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "validation"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [AccuracyCompletion]
    SUBJECTS = TYDIQA_LANGUAGES
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question"]
    LANGUAGE: dict[str, Language] | None = None

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        context = item.get("context", "")
        question = item.get("question", "")
        return f"{context}\nQuestion: {question}\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> list[str]:
        return [f" {a}" for a in item.get("answers", {}).get("text", [])]

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str]:
        return [f" {a}" for a in item.get("answers", {}).get("text", [])]
