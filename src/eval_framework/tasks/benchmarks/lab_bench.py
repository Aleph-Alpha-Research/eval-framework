import random
from typing import Any

from eval_framework.metrics.completion.accuracy_completion import AccuracyCompletion
from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.tasks.base import BaseTask, Language, ResponseType
from eval_framework.tasks.utils import get_n_letters

LAB_BENCH_SUBSETS = ["CloningScenarios", "DbQA", "FigQA", "LitQA2", "ProtocolQA", "SeqQA", "SuppQA", "TableQA"]


class LabBenchCloze(BaseTask[str]):
    NAME = "LabBenchCloze"
    DATASET_PATH = "futurehouse/lab-bench"
    SAMPLE_SPLIT = "train"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood, AccuracyCompletion]
    SUBJECTS = LAB_BENCH_SUBSETS
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question", "Answer"]
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        question = item.get("question", "")
        return f"Question: {question}\n"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        return f" {item.get('ideal')}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        choices = list(item.get("distractors", [])) + [item.get("ideal", "")]
        return [f" {c}" for c in choices]


class LabBenchMC(LabBenchCloze):
    NAME = "LabBenchMC"

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        question = item.get("question", "")
        choices = list(item.get("distractors", [])) + [item.get("ideal", "")]
        rng = random.Random(item.get("id", 0))
        order = list(range(len(choices)))
        rng.shuffle(order)
        shuffled_choices = [choices[i] for i in order]
        labels = get_n_letters(len(choices))
        options = "\n".join(f"{label}. {c}" for label, c in zip(labels, shuffled_choices))
        return f"Question: {question}\n{options}\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        choices = list(item.get("distractors", [])) + [item.get("ideal", "")]
        rng = random.Random(item.get("id", 0))
        order = list(range(len(choices)))
        rng.shuffle(order)
        labels = get_n_letters(len(choices))
        ideal_original_idx = len(choices) - 1
        gold_idx = order.index(ideal_original_idx)
        return f" {labels[gold_idx]}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        choices = list(item.get("distractors", [])) + [item.get("ideal", "")]
        labels = get_n_letters(len(choices))
        return [f" {label}" for label in labels]
