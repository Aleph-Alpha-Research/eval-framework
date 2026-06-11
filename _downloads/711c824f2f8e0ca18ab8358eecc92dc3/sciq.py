import hashlib
import random
from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.metrics.loglikelihood.bits_per_byte import BitsPerByteLoglikelihood
from eval_framework.metrics.loglikelihood.confidence_weighted_accuracy import ConfidenceWeightedAccuracy
from eval_framework.metrics.loglikelihood.dcs import DistributionalCorrectnessScore
from eval_framework.metrics.loglikelihood.ternary import TernaryScore
from eval_framework.tasks.base import NO_SUBJECT, BaseTask, Language, ResponseType
from eval_framework.tasks.utils import get_n_letters


def _shuffled_choices_and_correct_index(item: dict[str, Any]) -> tuple[list[str], int]:
    """Return (shuffled_choices, correct_index) with deterministic shuffle from item content."""
    choices = [
        item["distractor1"],
        item["distractor2"],
        item["distractor3"],
        item["correct_answer"],
    ]
    seed = int(hashlib.sha256((item["question"] + item["correct_answer"]).encode()).hexdigest(), 16)
    rng = random.Random(seed)
    order = list(range(4))
    rng.shuffle(order)
    shuffled = [choices[i] for i in order]
    correct_index = order.index(3)  # 3 = index of correct_answer in original list
    return shuffled, correct_index


class SCIQ(BaseTask[str]):
    """SciQ dataset: https://huggingface.co/datasets/allenai/sciq"""

    NAME = "SciQ"
    DATASET_PATH = "allenai/sciq"
    SAMPLE_SPLIT = "validation"  # 1000 examples (same split as lm-eval)
    FEWSHOT_SPLIT = "test"  # 1000 examples
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood, BitsPerByteLoglikelihood]
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
        shuffled, _ = _shuffled_choices_and_correct_index(item)
        return [f" {choice}" for choice in shuffled]


class SCIQ_OLMES(SCIQ):
    """
    SciQ with OLMES-style prompt: options shown with space-prefixed labels (" A.", " B.", " C.", " D.");
    loglikelihood over " A"/" B"/" C"/" D". Answer choices are deterministically shuffled per example.
    """

    NAME = "SciQ_OLMES"
    SAMPLE_SPLIT = "train"  # Use train split (largest) to best match OLMES, which evaluates all splits
    FEWSHOT_SPLIT = "train"

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        self.keys = get_n_letters(4)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        question = item["question"]
        shuffled, _ = _shuffled_choices_and_correct_index(item)
        options = "\n".join(f" {key}. {choice}" for key, choice in zip(self.keys, shuffled))
        return f"Question: {question}\n{options}\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        _, correct_index = _shuffled_choices_and_correct_index(item)
        return f" {self.keys[correct_index]}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [f" {key}" for key in self.keys]


class SCIQ_IDK(SCIQ):
    NAME = "SciQ_IDK"
    METRICS = [
        AccuracyLoglikelihood,
        AccuracyNormLoglikelihood,
        ConfidenceWeightedAccuracy,
        DistributionalCorrectnessScore,
        TernaryScore,
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
        TernaryScore,
    ]

    def _get_initial_prompt_text(self, item: dict[str, Any]) -> str:
        return (
            "Answer only if you are confident, since mistakes may be penalised, while correct answers receive points. "
            "It is acceptable to answer with 'don't know' if you are unsure, and you will receive 0 points."
        )

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        completions = super()._get_possible_completions(item)
        return (completions or []) + [" don't know"]
