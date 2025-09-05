from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.tasks.base import BaseTask, Language, ResponseType
from eval_framework.tasks.utils import get_n_letters


class BELEBELE(BaseTask[str]):
    """BELEBELE dataset: https://huggingface.co/datasets/facebook/belebele"""

    NAME = "BELEBELE"
    DATASET_PATH = "facebook/belebele"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "test"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
    SUBJECTS = [
        "eng_Latn",
    ]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question", "Answer"] + get_n_letters(4)
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int, custom_subjects: list[str] | None, custom_hf_revision: str | None) -> None:
        super().__init__(num_fewshot, custom_subjects, custom_hf_revision)

        self.keys = get_n_letters(4)
        self.num_to_letter = {str(i): letter for i, letter in enumerate(self.keys, start=1)}

    def _get_initial_prompt_text(self, item: dict[str, Any]) -> str:
        return "The following are multiple choice questions (with answers)."

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        context = item["flores_passage"].strip()
        question = item["question"].strip()
        choices = "".join(
            [
                f"{key}. {choice}\n"
                for key, choice in zip(
                    self.keys, [item["mc_answer1"], item["mc_answer2"], item["mc_answer3"], item["mc_answer4"]]
                )
            ]
        )
        return f"{context}\n\nQuestion: {question}\n{choices}"

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        ground_truth = self._get_ground_truth(item)
        assert ground_truth is not None
        return f"{self._get_cue_text(item)}{ground_truth}"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        return f" {self.keys[int(item['correct_answer_num']) - 1]}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [f" {key}" for key in self.keys]
