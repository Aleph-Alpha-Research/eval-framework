from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.metrics.loglikelihood.confidence_weighted_accuracy import ConfidenceWeightedAccuracy
from eval_framework.metrics.loglikelihood.distributional_correctness_score import DistributionalCorrectnessScore
from eval_framework.metrics.loglikelihood.ternary import TernaryScore
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
        return f"Question: {item['question']}\n"

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


class SCIQ_IDK(SCIQ):
    NAME = "SciQ_IDK"
    METRICS = [
        AccuracyLoglikelihood,
        AccuracyNormLoglikelihood,
        ConfidenceWeightedAccuracy,
        DistributionalCorrectnessScore,
        TernaryScore
    ]

    def _get_initial_prompt_text(self, item: dict[str, Any]) -> str:
        return (
            "Answer only if you are confident, since mistakes may be penalised, while correct answers receive points. "
            "It is acceptable to answer with 'don't know' if you are unsure, and you will receive 0 points."
        )

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        completions = super()._get_possible_completions(item)
        return (completions or []) + [" don't know"]


class SCIQEvalHarness(SCIQ):
    """Based on
    https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/sciq/sciq.yaml#L8
    In the Eval Harness implementation, the instruction text includes a context passage.
    This passage often contains the answer, reducing the benchmark to a straightforward copy-and-paste task.
    """

    NAME = "SciQ Eval Harness"
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


class SCIQEvalHarness_IDK(SCIQEvalHarness):
    NAME = "SciQ Eval Harness_IDK"
    METRICS = [
        AccuracyLoglikelihood,
        AccuracyNormLoglikelihood,
        ConfidenceWeightedAccuracy,
        DistributionalCorrectnessScore,
        TernaryScore
    ]

    def _get_initial_prompt_text(self, item: dict[str, Any]) -> str:
        return (
            "Answer only if you are confident, since mistakes may be penalised, while correct answers receive points. "
            "It is acceptable to answer with 'don't know' if you are unsure, and you will receive 0 points."
        )

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        completions = super()._get_possible_completions(item)
        return (completions or []) + [" don't know"]
