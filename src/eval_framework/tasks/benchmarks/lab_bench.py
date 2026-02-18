import random
from typing import Any

from eval_framework.metrics.completion.accuracy_completion import AccuracyCompletion
from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.metrics.loglikelihood.bits_per_byte import BitsPerByteLoglikelihood
from eval_framework.tasks.base import BaseTask, Language, ResponseType
from eval_framework.tasks.utils import get_n_letters

LAB_BENCH_SUBSETS = ["CloningScenarios", "DbQA", "FigQA", "LitQA2", "ProtocolQA", "SeqQA", "SuppQA", "TableQA"]


class LabBenchCloze(BaseTask[str]):
    """Lab-Bench (futurehouse/lab-bench): QA over scientific protocols; cloze ranks ideal vs distractors."""

    NAME = "LabBenchCloze"
    DATASET_PATH = "futurehouse/lab-bench"
    SAMPLE_SPLIT = "train"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood, AccuracyCompletion, BitsPerByteLoglikelihood]
    SUBJECTS = LAB_BENCH_SUBSETS
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question", "Answer"]
    LANGUAGE = Language.ENG

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        question = item.get("question", "")
        return f"Question: {question}\n"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        ideal = item.get("ideal")
        if ideal is None:
            return None
        return f" {ideal}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        choices = list(item.get("distractors", [])) + [item.get("ideal", "")]
        return [f" {c}" for c in choices]


class LabBenchMC(LabBenchCloze):
    NAME = "LabBenchMC"

    def _get_choices_order_keys(self, item: dict[str, Any]) -> tuple[list[str], list[int], list[str]]:
        """Return (choices, shuffle_order, keys) for consistent ordering across methods."""
        choices = list(item.get("distractors", [])) + [item.get("ideal", "")]
        rng = random.Random(item.get("id", 0))
        order = list(range(len(choices)))
        rng.shuffle(order)
        keys = get_n_letters(len(choices))
        return choices, order, keys

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        question = item.get("question", "")
        choices, order, keys = self._get_choices_order_keys(item)
        shuffled_choices = [choices[i] for i in order]
        options = "\n".join(f" {key}. {c}" for key, c in zip(keys, shuffled_choices))
        return f"Question: {question}\n{options}\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        choices, order, keys = self._get_choices_order_keys(item)
        ideal_original_idx = len(choices) - 1
        gold_idx = order.index(ideal_original_idx)
        return f" {keys[gold_idx]}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        _, _, keys = self._get_choices_order_keys(item)
        return [f" {label}" for label in keys]


class LabBenchMC_OLMES(LabBenchMC):
    """
    LabBenchMC with OLMES-style prompt: space before each label in the prompt (" A.", " B.", ...).
    """

    NAME = "LabBenchMC_OLMES"

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        question = item.get("question", "")
        choices, order, keys = self._get_choices_order_keys(item)
        shuffled_choices = [choices[i] for i in order]
        options = "\n".join(f" {key}. {c}" for key, c in zip(keys, shuffled_choices))
        return f"Question: {question}\n{options}\n"
