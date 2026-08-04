"""German ARC (EllaMind) tasks.

https://huggingface.co/datasets/ellamind/arc-multilingual
"""

from typing import Any

from eval_framework.tasks.base import BaseTask, Language
from eval_framework.tasks.dataset_revisions import HF_REVISIONS_LOCKFILE
from eval_framework.tasks.task_style import BPBStyle, ClozeStyle, MCStyle, answer_key_to_index
from eval_framework.tasks.utils import get_n_letters


class _ARC_ELLAMIND_DE_Base(BaseTask[str]):
    """Non-registered base for all German ARC (EllaMind) variants.

    Dataset: https://huggingface.co/datasets/ellamind/arc-multilingual

    The HuggingFace config (``name``) is the language, so we always load the German
    subset (``deu``). To mirror regular ARC behavior, we expose ARC subsets via
    ``SUBJECTS`` (``ARC-Easy`` / ``ARC-Challenge``) and filter rows by the
    ``arc_config`` column after loading.
    """

    DATASET_PATH = "ellamind/arc-multilingual"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "test"
    SUBJECTS = ["ARC-Easy", "ARC-Challenge"]
    LANGUAGE = Language.DEU
    PERTURBATION_UNMODIFIABLE_WORDS = ["Frage", "Answer"] + get_n_letters(5)
    _LANGUAGE_SUBSET = "deu"

    def _load_dataset(self, subject: str) -> None:
        # Subject encodes the ARC subset, while HF config encodes the language.
        super()._load_dataset(self._LANGUAGE_SUBSET)
        self.dataset = {
            split: [item for item in items if item["arc_config"] == subject] for split, items in self.dataset.items()
        }

    def _get_raw_question(self, item: dict[str, Any]) -> str:
        return item["question"]

    def _get_choices(self, item: dict[str, Any]) -> list[str]:
        return item["choices"]

    def _get_correct_index(self, item: dict[str, Any]) -> int:
        return answer_key_to_index(item["answer_key"])


class ARC_ELLAMIND_CLOZE_DE(_ARC_ELLAMIND_DE_Base):
    """German ARC - Cloze format."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "ARC_ELLAMIND_CLOZE_DE"
    TASK_STYLER = ClozeStyle().for_language(Language.DEU)


class ARC_ELLAMIND_MC_DE(_ARC_ELLAMIND_DE_Base):
    """German ARC - Multiple-choice format."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "ARC_ELLAMIND_MC_DE"
    TASK_STYLER = MCStyle().for_language(Language.DEU)


class ARC_ELLAMIND_BPB_DE(_ARC_ELLAMIND_DE_Base):
    """German ARC - BPB format."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "ARC_ELLAMIND_BPB_DE"
    TASK_STYLER = BPBStyle().for_language(Language.DEU)
