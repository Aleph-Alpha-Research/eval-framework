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
    SUBJECTS = ["additional"]
    PERTURBATION_UNMODIFIABLE_WORDS = get_n_letters(4)
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        self.keys = get_n_letters(4)
        self.num_to_letter = {str(i): letter for i, letter in enumerate(self.keys, start=1)}

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        question = item["question_stem"].strip()
        fact = item["fact1"].strip()
        choices = "".join([f"{choice.strip()}\n" for key, choice in zip(self.keys, item["choices"]["text"])])
        return f"Fact: {fact}\nComplete: {question}:\n{choices}"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        answer_key = self.num_to_letter.get(item["answerKey"], item["answerKey"])
        return f" {item['choices']['text'][self.keys.index(answer_key)].strip()}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [f" {choice}" for choice in item["choices"]["text"]]

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        ground_truth = self._get_ground_truth(item)
        assert ground_truth is not None
        return f"{self._get_cue_text(item)}{ground_truth}"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"


class OPENBOOKQA_EVAL_HARNESS(OPENBOOKQA):
    """Closed-book version of OpenBookQA — question only, no supporting fact."""

    NAME = "OpenBookQA_ClosedBook"

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        question = item["question_stem"].strip()
        choices = "".join([f"{choice.strip()}\n" for key, choice in zip(self.keys, item["choices"]["text"])])
        return f"Complete: {question}:\n{choices}"
