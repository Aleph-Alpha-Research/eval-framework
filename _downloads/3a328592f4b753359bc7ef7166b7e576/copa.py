from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.metrics.loglikelihood.confidence_weighted_accuracy import ConfidenceWeightedAccuracy
from eval_framework.metrics.loglikelihood.dcs import DistributionalCorrectnessScore
from eval_framework.metrics.loglikelihood.ternary import TernaryScore
from eval_framework.tasks.base import BaseTask, Language, ResponseType
from eval_framework.tasks.utils import get_n_letters


class COPAEvalHarness(BaseTask[str]):
    """COPA dataset: https://huggingface.co/datasets/aps/super_glue
    This version uses samples from the validation split as evaluation examples (same as lm-eval-harness).
    """

    NAME = "COPAEvalHarness"
    DATASET_PATH = "aps/super_glue"
    SAMPLE_SPLIT = "validation"  # 100 examples (same split as lm-eval)
    FEWSHOT_SPLIT = "test"  # 500 examples
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
    SUBJECTS = ["copa"]
    PERTURBATION_UNMODIFIABLE_WORDS = ["because", "therefore"]
    LANGUAGE = Language.ENG

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        connector = {
            "cause": "because",
            "effect": "therefore",
        }[item["question"]]
        return item["premise"].strip()[:-1] + f" {connector} "

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        correct_choice = item["choice1"] if item["label"] == 0 else item["choice2"]
        return f"{self.convert_choice(correct_choice)}"

    def convert_choice(self, choice: str) -> str:
        return choice[0].lower() + choice[1:]

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        choices = [self.convert_choice(item["choice1"]), self.convert_choice(item["choice2"])]
        return choices


class COPA_OLMES(COPAEvalHarness):
    """
    COPA multiple choice (OLMES/oe_eval style): prompt shows premise + connector and options with
    space-prefixed labels (" A.", " B."); loglikelihood over " A"/" B".
    """

    NAME = "COPA_OLMES"

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        connector = {
            "cause": "because",
            "effect": "therefore",
        }[item["question"]]
        premise = item["premise"].strip()[:-1] + f" {connector}"
        choices = [self.convert_choice(item["choice1"]), self.convert_choice(item["choice2"])]
        labels = get_n_letters(len(choices))
        options = "\n".join(f" {label}. {choice}" for label, choice in zip(labels, choices))
        return f"{premise}\n{options}\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        labels = get_n_letters(2)
        return f" {labels[item['label']]}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [f" {label}" for label in get_n_letters(2)]


class COPA(COPAEvalHarness):
    """
    Unlike the original COPA task, this version uses the test split for evaluation and the validation split for
    few-shot examples. Previously, the test split labels were unavailable in the original dataset, but they are
    now accessible, allowing this configuration.
    """

    NAME = "COPA"
    SAMPLE_SPLIT = "test"  # 500 examples
    FEWSHOT_SPLIT = "validation"  # 100 examples


class COPA_IDKEvalHarness(COPAEvalHarness):
    NAME = "COPA_IDKEvalHarness"
    METRICS = [
        AccuracyLoglikelihood,
        AccuracyNormLoglikelihood,
        ConfidenceWeightedAccuracy,
        DistributionalCorrectnessScore,
        TernaryScore,
    ]

    def _get_initial_prompt_text(self, item: dict[str, Any]) -> str:
        return (
            "Complete the sentence only if you are confident, since mistakes may be penalised, while correct "
            "answers receive points. It is acceptable to answer with 'I do not know' if you are unsure, and "
            "you will receive 0 points."
        )

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        completions = super()._get_possible_completions(item)
        return (completions or []) + [" I do not know."]


class COPA_IDK(COPA_IDKEvalHarness):
    NAME = "COPA_IDK"
    SAMPLE_SPLIT = "test"  # 500 examples
    FEWSHOT_SPLIT = "validation"  # 100 examples
