from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.tasks.base import BaseTask, Language, ResponseType
from eval_framework.tasks.utils import get_n_letters


class ARC_FI(BaseTask[str]):
    """ARC-FI dataset: https://huggingface.co/datasets/LumiOpen/arc_challenge_mt"""

    NAME = "ARC Finnish"
    DATASET_PATH = "LumiOpen/arc_challenge_mt"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "validation"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
    SUBJECTS = ["fi"]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question"] + get_n_letters(5)
    LANGUAGE = Language.FIN

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)

        self.keys = get_n_letters(5)  # needs to be 5 because there is one sample with 5 answer possibilities
        self.num_to_letter = {str(i): letter for i, letter in enumerate(self.keys, start=1)}

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"Question: {item['question']}\n"

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        ground_truth = self._get_ground_truth(item)
        assert ground_truth is not None
        return f"{self._get_cue_text(item)}{ground_truth}"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        answer_key = self.num_to_letter.get(item["answerKey"], item["answerKey"])
        return f" {item['choices']['text'][self.keys.index(answer_key)]}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [f" {choice}" for choice in item["choices"]["text"]]
