from typing import Any

from eval_framework.metrics.completion.drop_completion import DropF1ExactMatch, DropMetricContext
from eval_framework.tasks.base import NO_SUBJECT, BaseTask, Language, ResponseType
from eval_framework.tasks.task_style import (
    BPBStyle,
    ClozeStyle,
    MCStyle,
    answer_key_to_index,
)


class NaturalQsOpen(BaseTask[str]):
    NAME = "NaturalQsOpen"
    DATASET_PATH = "google-research-datasets/nq_open"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [DropF1ExactMatch]
    SUBJECTS = [NO_SUBJECT]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question", "Answer"]
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        self.stop_sequences = ["Question:", "Q:", "\n\n"]
        self.max_tokens = 50

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"Question: {item.get('question', '')}\n"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_ground_truth(self, item: dict[str, Any]) -> list[str]:
        return [f" {a}" for a in item.get("answer", [])]

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        ground_truths = self._get_ground_truth(item)
        assert ground_truths is not None

        # Extra processing step, since for this task, the ground truth can be a list of strings.
        # Following, OLMES, we join the targets with a comma.
        # TODO: Explore other ways (e.g. select a single target). The correct approach depends on the question.
        # E.g. "how many seasons of vampire diaries r there" [ "eight", "8" ] should perhaps select just one.
        # but "what are the three fifty shades of grey books"
        # [ "Fifty Shades of Grey", "Fifty Shades Darker", "Fifty Shades Freed" ]
        # would be better with joining.
        target = ",".join(ground_truths)  # only comma, since the targets are already space-separated.

        assert isinstance(target, str)
        return f"{self._get_cue_text(item)}{target}"

    def _get_context(self, item: dict[str, Any]) -> DropMetricContext | None:
        # DROP metric expects list of lists of strings.
        answers = item.get("answer", [])
        if not answers:
            return None
        return DropMetricContext(answer_tuples=[[a] for a in answers])


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
    TASK_STYLER = ClozeStyle()


class NaturalQsOpenMC(_NaturalQsOpenChoice_Base):
    NAME = "NaturalQsOpenMC"
    TASK_STYLER = MCStyle(space_prefixed_labels=True)


class NaturalQsOpenMC_OLMES(_NaturalQsOpenChoice_Base):
    """NaturalQsOpenMC with OLMES-style prompt: space before each label in the prompt (" A.", " B.", ...)."""

    NAME = "NaturalQsOpenMC_OLMES"
    TASK_STYLER = MCStyle(space_prefixed_labels=True)


class NaturalQsOpenBPB(_NaturalQsOpenChoice_Base):
    """BPB-only variant."""

    NAME = "NaturalQsOpenBPB"
    TASK_STYLER = BPBStyle()
