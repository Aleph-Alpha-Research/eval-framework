from typing import Any

from eval_framework.metrics.completion.accuracy_completion import AccuracyCompletion
from eval_framework.metrics.completion.f1 import F1
from eval_framework.tasks.base import NO_SUBJECT, BaseTask, Language, ResponseType
from eval_framework.tasks.formatting import (
    ClozeFormatter,
    MCFormatter,
    answer_key_to_index,
)


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


class _NaturalQsOpenChoice_Base(BaseTask[str]):
    """Shared base for choice-based NaturalQsOpen variants (Cloze, MC, MC_OLMES)."""

    DATASET_PATH = "allenai/nq-gen2mc"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "validation"
    SUBJECTS = [NO_SUBJECT]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question", "Answer"]
    LANGUAGE = Language.ENG

    def _get_raw_question(self, item: dict[str, Any]) -> str:
        return item.get("question", "")

    def _get_choices(self, item: dict[str, Any]) -> list[str]:
        return item.get("choices", {}).get("text", [])

    def _get_correct_index(self, item: dict[str, Any]) -> int:
        return answer_key_to_index(item.get("answerKey", ""))


class NaturalQsOpenCloze(_NaturalQsOpenChoice_Base):
    NAME = "NaturalQsOpenCloze"
    TASK_STYLER = ClozeFormatter()


class NaturalQsOpenMC(_NaturalQsOpenChoice_Base):
    NAME = "NaturalQsOpenMC"
    TASK_STYLER = MCFormatter(space_prefixed_labels=True)


class NaturalQsOpenMC_OLMES(_NaturalQsOpenChoice_Base):
    """NaturalQsOpenMC with OLMES-style prompt: space before each label in the prompt (" A.", " B.", ...)."""

    NAME = "NaturalQsOpenMC_OLMES"
    TASK_STYLER = MCFormatter(space_prefixed_labels=True)
