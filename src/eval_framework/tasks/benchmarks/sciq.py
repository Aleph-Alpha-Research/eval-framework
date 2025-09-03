from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.tasks.base import NO_SUBJECT, BaseTask, Language, ResponseType


class SCIQ(BaseTask[str]):
    """SciQ dataset: https://huggingface.co/datasets/allenai/sciq"""

    NAME = "SciQ"
    DATASET_PATH = "allenai/sciq"
    SAMPLE_SPLIT = "validation"  # 1000 examples (same split as lm-eval)
    FEWSHOT_SPLIT = "test"  # 1000 examples
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
    SUBJECTS = [NO_SUBJECT]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question"]
    LANGUAGE = Language.ENG

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"{item['support'].lstrip()}\nQuestion: {item['question']}\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        return f" {item['correct_answer']}"

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        ground_truth = self._get_ground_truth(item)
        assert ground_truth is not None
        return f"{self._get_cue_text(item)}{ground_truth}"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        choices = [
            item["distractor1"],
            item["distractor2"],
            item["distractor3"],
            item["correct_answer"],
        ]
        return [f" {choice}" for choice in choices]
