import re
from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.metrics.loglikelihood.confidence_weighted_accuracy import ConfidenceWeightedAccuracy
from eval_framework.metrics.loglikelihood.distributional_correctness_score import DistributionalCorrectnessScore
from eval_framework.metrics.loglikelihood.ternary import TernaryScore
from eval_framework.tasks.base import NO_SUBJECT, BaseTask, Language, ResponseType


class HELLASWAG(BaseTask[str]):
    """Hellaswag dataset: https://huggingface.co/datasets/Rowan/hellaswag
    available data set sections: train, validation, test"""

    NAME = "HellaSwag"
    DATASET_PATH = "Rowan/hellaswag"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
    SUBJECTS = [NO_SUBJECT]
    LANGUAGE = Language.ENG

    @staticmethod
    def _preprocess(prompt: str) -> str:
        # remove bracketed text
        prompt = prompt.strip()
        prompt = prompt.replace(" [title]", ". ")
        prompt = re.sub("\\[.*?\\]", "", prompt)
        prompt = prompt.replace("  ", " ")
        return prompt

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        subject = self._preprocess(item["activity_label"])
        question = self._preprocess(item["ctx_a"] + " " + item["ctx_b"].capitalize()).strip()
        return f"{subject}: {question}"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        ground_truth_index = int(item["label"] if item["label"] != "" else 0)
        choices = [self._preprocess(ending) for ending in item["endings"]]
        return f" {choices[ground_truth_index]}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [f" {self._preprocess(ending)}" for ending in item["endings"]]

class HELLASWAG_DCS(HELLASWAG):
    NAME = "HellaSwag_DCS"
    METRICS = [
        AccuracyLoglikelihood,
        AccuracyNormLoglikelihood,
        ConfidenceWeightedAccuracy,
        DistributionalCorrectnessScore,
        TernaryScore
    ]

    def _get_initial_prompt_text(self, item: dict[str, Any]) -> str:
        return (
            "Complete the sentence only if you are confident, since mistakes may be penalised, while correct completions receive points. "
            "It is acceptable to answer with 'I do not know' if you are unsure, and you will receive 0 points."
        )

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        completions = super()._get_possible_completions(item)
        return (completions or []) + [" I do not know."]