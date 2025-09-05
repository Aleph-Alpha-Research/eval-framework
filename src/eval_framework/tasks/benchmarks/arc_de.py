from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.tasks.base import NO_SUBJECT, BaseTask, Language, ResponseType
from eval_framework.tasks.utils import get_n_letters


class ARC_DE(BaseTask[str]):
    """ARC-DE dataset: https://huggingface.co/datasets/LeoLM/ArcChallenge_de"""

    NAME = "ARC German"
    DATASET_PATH = "LeoLM/ArcChallenge_de"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "validation"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
    SUBJECTS = [NO_SUBJECT]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Frage"] + get_n_letters(5)
    LANGUAGE = Language.DEU

    def __init__(self, num_fewshot: int, custom_subjects: list[str] | None, custom_hf_revision: str | None) -> None:
        super().__init__(num_fewshot, custom_subjects, custom_hf_revision)

        self.keys = get_n_letters(5)  # needs to be 5 because there is one sample with 5 answer possibilities
        self.num_to_letter = {str(i): letter for i, letter in enumerate(self.keys, start=1)}

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"Frage: {item['question_de']}\n"

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        ground_truth = self._get_ground_truth(item)
        assert ground_truth is not None
        return f"{self._get_cue_text(item)}{ground_truth}"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Antwort:"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        answer_key = self.num_to_letter.get(item["answerKey"], item["answerKey"])
        return f" {item['choices_de']['text'][self.keys.index(answer_key)]}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [f" {choice}" for choice in item["choices_de"]["text"]]
