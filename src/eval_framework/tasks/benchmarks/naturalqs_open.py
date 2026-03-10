from typing import Any

from eval_framework.metrics.completion.accuracy_completion import AccuracyCompletion
from eval_framework.metrics.completion.f1 import F1
from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.metrics.loglikelihood.bits_per_byte import BitsPerByteLoglikelihood
from eval_framework.tasks.base import NO_SUBJECT, BaseTask, Language, ResponseType
from eval_framework.tasks.utils import get_n_letters


class NaturalQsOpen(BaseTask[str]):
    NAME = "NaturalQsOpen"
    DATASET_PATH = "google-research-datasets/nq_open"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [AccuracyCompletion, F1]
    SUBJECTS = [NO_SUBJECT]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question", "Answer"]
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        self.stop_sequences = ["Question:", "Q:", "\n\n"]
        self.max_tokens = 50

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"Question: {item.get('question', '')}"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_ground_truth(self, item: dict[str, Any]) -> list[str]:
        return [f" {a}" for a in item.get("answer", [])]

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        ground_truth = self._get_ground_truth(item)
        assert ground_truth is not None
        return f"{self._get_cue_text(item)}{ground_truth}"


class NaturalQsOpenCloze(BaseTask[str]):
    NAME = "NaturalQsOpenCloze"
    DATASET_PATH = "allenai/nq-gen2mc"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "validation"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood, BitsPerByteLoglikelihood]
    SUBJECTS = [NO_SUBJECT]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question", "Answer"]
    LANGUAGE = Language.ENG

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"Question: {item.get('question', '')}\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        texts = item.get("choices", {}).get("text", [])
        labels = item.get("choices", {}).get("label", [])
        gold_idx = labels.index(item.get("answerKey", ""))
        return f" {texts[gold_idx]}"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        texts = item.get("choices", {}).get("text", [])
        return [f" {t}" for t in texts]

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        ground_truth = self._get_ground_truth(item)
        assert ground_truth is not None
        return f"{self._get_cue_text(item)}{ground_truth}"


class NaturalQsOpenMC(NaturalQsOpenCloze):
    NAME = "NaturalQsOpenMC"

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        self.keys = get_n_letters(4)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        question = item.get("question", "")
        texts = item.get("choices", {}).get("text", [])
        options = "\n".join(f" {key}. {t}" for key, t in zip(self.keys, texts))
        return f"Question: {question}\n{options}\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        gold_idx = self.keys.index(item.get("answerKey", ""))
        return f" {self.keys[gold_idx]}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [f" {key}" for key in self.keys]

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        ground_truth = self._get_ground_truth(item)
        assert ground_truth is not None
        return f"{self._get_cue_text(item)}{ground_truth}"


class NaturalQsOpenMC_OLMES(NaturalQsOpenMC):
    """
    NaturalQsOpenMC with OLMES-style prompt: space before each label in the prompt (" A.", " B.", ...).
    """

    NAME = "NaturalQsOpenMC_OLMES"

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        question = item.get("question", "")
        texts = item.get("choices", {}).get("text", [])
        options = "\n".join(f" {key}. {t}" for key, t in zip(self.keys, texts))
        return f"Question: {question}\n{options}\n"
