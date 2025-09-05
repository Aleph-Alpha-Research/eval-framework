from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.tasks.base import BaseTask, Language, ResponseType
from eval_framework.tasks.utils import get_n_letters


class OPENBOOKQA(BaseTask[str]):
    """OpenBookQA dataset: https://huggingface.co/datasets/allenai/openbookqa"""

    NAME = "OpenBookQA"
    DATASET_PATH = "allenai/openbookqa"
    SAMPLE_SPLIT = "validation"  # 500 examples (same split as lm-eval)
    FEWSHOT_SPLIT = "test"  # 500 examples
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
    SUBJECTS = ["main"]
    PERTURBATION_UNMODIFIABLE_WORDS = get_n_letters(4)
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int, custom_subjects: list[str] | None, custom_hf_revision: str | None) -> None:
        super().__init__(num_fewshot, custom_subjects, custom_hf_revision)
        self.keys = get_n_letters(4)
        self.num_to_letter = {str(i): letter for i, letter in enumerate(self.keys, start=1)}

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"{item['question_stem'].strip()}"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        answer_key = self.num_to_letter.get(item["answerKey"], item["answerKey"])
        return f" {item['choices']['text'][self.keys.index(answer_key)].strip()}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [f" {choice}" for choice in item["choices"]["text"]]
