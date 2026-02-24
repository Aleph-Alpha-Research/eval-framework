from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.metrics.loglikelihood.bits_per_byte import BitsPerByteLoglikelihood
from eval_framework.tasks.base import NO_SUBJECT, BaseTask, Language, ResponseType
from eval_framework.tasks.utils import get_n_letters


class CommonsenseQACloze(BaseTask[str]):
    """CommonsenseQA dataset: https://huggingface.co/datasets/tau/commonsense_qa"""

    NAME = "CommonsenseQACloze"
    DATASET_PATH = "tau/commonsense_qa"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "validation"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
    SUBJECTS = [NO_SUBJECT]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question"]
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        self.keys = get_n_letters(5)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"Question: {item['question']}\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        correct_label = item["answerKey"]
        correct_index = self.keys.index(correct_label)
        return f" {self.keys[correct_index]}"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [f" {choice}" for choice in item["choices"]["text"]]


class CommonsenseQAFullTextCloze(CommonsenseQACloze):
    """
    CommonsenseQA cloze with full answer text as ground truth (not just the letter).
    Scores loglikelihood over the full correct choice text; includes bits-per-byte.
    """

    NAME = "CommonsenseQAFullTextCloze"
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood, BitsPerByteLoglikelihood]

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        correct_label = item["answerKey"]
        correct_index = self.keys.index(correct_label)
        return f" {item['choices']['text'][correct_index]}"


class CommonsenseQAMC(CommonsenseQACloze):
    """Multiple-choice variant of CommonsenseQA where the model selects a letter (A-E)."""

    NAME = "CommonsenseQAMC"

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        question = item["question"]
        texts = item["choices"]["text"]
        options = "\n".join(f" {key}. {choice}" for key, choice in zip(self.keys, texts))
        return f"Question: {question}\n{options}\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        correct_label = item["answerKey"]
        return f" {correct_label}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        labels = item["choices"]["label"]
        return [f" {label}" for label in labels]


class CommonsenseQAMC_OLMES(CommonsenseQAMC):
    """
    CommonsenseQA MC with OLMES-style prompt: space before each label in the prompt (" A.", " B.", ...).
    """

    NAME = "CommonsenseQAMC_OLMES"
    FEWSHOT_SPLIT = "train"

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        question = item["question"]
        texts = item["choices"]["text"]
        options = "\n".join(f" {key}. {choice}" for key, choice in zip(self.keys, texts))
        return f"Question: {question}\n{options}\n"
