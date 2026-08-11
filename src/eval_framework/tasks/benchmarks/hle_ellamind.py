"""German HLE (Humanity's Last Exam, EllaMind) tasks.

https://huggingface.co/datasets/ellamind/hle-multilingual

HLE uses a single distractor set (``incorrect_answers``). The natively
multiple-choice subset is exposed via ``_NATIVE_MC_ONLY = True`` on the subclass.
"""

from typing import Any

from eval_framework.tasks.base import BaseTask, Language
from eval_framework.tasks.dataset_revisions import HF_REVISIONS_LOCKFILE
from eval_framework.tasks.task_style import BPBStyle, ClozeStyle, MCStyle, shuffle_correct_with_distractors


class _HLE_ELLAMIND_DE_Base(BaseTask[str]):
    """Non-registered base for German HLE (EllaMind) variants.

    Dataset: https://huggingface.co/datasets/ellamind/hle-multilingual

    Set ``_NATIVE_MC_ONLY = True`` to restrict to the ~half of items that are
    natively multiple-choice in the original benchmark.
    """

    DATASET_PATH = "ellamind/hle-multilingual"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "test"
    SUBJECTS = ["deu"]
    LANGUAGE = Language.DEU
    _NATIVE_MC_ONLY: bool = False

    def _load_dataset(self, subject: str) -> None:
        super()._load_dataset(subject)
        if self._NATIVE_MC_ONLY:
            self.dataset = {
                split: [item for item in items if item["answer_type"] == "multipleChoice"]
                for split, items in self.dataset.items()
            }

    def _shuffled(self, item: dict[str, Any]) -> tuple[list[str], int]:
        return shuffle_correct_with_distractors(
            correct=item["correct_answer"],
            distractors=item["incorrect_answers"],
            seed_text=item["question"] + item["correct_answer"],
        )

    def _get_raw_question(self, item: dict[str, Any]) -> str:
        return item["question"]

    def _get_choices(self, item: dict[str, Any]) -> list[str]:
        return self._shuffled(item)[0]

    def _get_correct_index(self, item: dict[str, Any]) -> int:
        return self._shuffled(item)[1]


class HLE_ELLAMIND_MC_DE(_HLE_ELLAMIND_DE_Base):
    """German HLE - MC format (all 800 items)."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "HLE_ELLAMIND_MC_DE"
    TASK_STYLER = MCStyle().for_language(Language.DEU)


class HLE_ELLAMIND_CLOZE_DE(_HLE_ELLAMIND_DE_Base):
    """German HLE - Cloze format (all 800 items)."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "HLE_ELLAMIND_CLOZE_DE"
    TASK_STYLER = ClozeStyle().for_language(Language.DEU)


class HLE_ELLAMIND_MC_NATIVE_DE(_HLE_ELLAMIND_DE_Base):
    """German HLE - MC format, native multiple-choice items only."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "HLE_ELLAMIND_MC_NATIVE_DE"
    _NATIVE_MC_ONLY = True
    TASK_STYLER = MCStyle().for_language(Language.DEU)


class HLE_ELLAMIND_CLOZE_NATIVE_DE(_HLE_ELLAMIND_DE_Base):
    """German HLE - Cloze format, native multiple-choice items only."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "HLE_ELLAMIND_CLOZE_NATIVE_DE"
    _NATIVE_MC_ONLY = True
    TASK_STYLER = ClozeStyle().for_language(Language.DEU)


class HLE_ELLAMIND_BPB_DE(_HLE_ELLAMIND_DE_Base):
    """German HLE - BPB format (all 800 items)."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "HLE_ELLAMIND_BPB_DE"
    TASK_STYLER = BPBStyle().for_language(Language.DEU)
