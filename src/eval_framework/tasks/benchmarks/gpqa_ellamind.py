"""German GPQA (Graduate-level Professional QA, EllaMind) tasks.

https://huggingface.co/datasets/ellamind/gpqa-multilingual

GPQA uses a single distractor set (``incorrect_answers``). Its diamond subset (the 198
hardest questions) is exposed via ``_DIAMOND_ONLY = True`` on the subclass.
"""

from typing import Any

from eval_framework.tasks.base import BaseTask, Language
from eval_framework.tasks.dataset_revisions import HF_REVISIONS_LOCKFILE
from eval_framework.tasks.task_style import BPBStyle, ClozeStyle, MCStyle, shuffle_correct_with_distractors


class _GPQA_ELLAMIND_DE_Base(BaseTask[str]):
    """Non-registered base for German GPQA (EllaMind) variants.

    Dataset: https://huggingface.co/datasets/ellamind/gpqa-multilingual

    The diamond subset (``is_diamond=True``, 198 of 448 items) is the hardest
    questions from the original GPQA-Diamond benchmark. Set ``_DIAMOND_ONLY =
    True`` on a subclass to restrict evaluation to that subset.

    Note, we don't use `domain` or `subdomain` fields at the moment (they are not translated).
    Also, we don't use an initial system prompt, different to the base GPQA task.
    """

    DATASET_PATH = "ellamind/gpqa-multilingual"
    SAMPLE_SPLIT = "train"
    FEWSHOT_SPLIT = "train"
    SUBJECTS = ["deu"]
    LANGUAGE = Language.DEU
    _DIAMOND_ONLY: bool = False

    def _load_dataset(self, subject: str) -> None:
        super()._load_dataset(subject)
        if self._DIAMOND_ONLY:
            self.dataset = {
                split: [item for item in items if item["is_diamond"]] for split, items in self.dataset.items()
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


class GPQA_ELLAMIND_MC_DE(_GPQA_ELLAMIND_DE_Base):
    """German GPQA - MC format (all 448 items)."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "GPQA_ELLAMIND_MC_DE"
    TASK_STYLER = MCStyle().for_language(Language.DEU)


class GPQA_ELLAMIND_CLOZE_DE(_GPQA_ELLAMIND_DE_Base):
    """German GPQA - Cloze format (all 448 items)."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "GPQA_ELLAMIND_CLOZE_DE"
    TASK_STYLER = ClozeStyle().for_language(Language.DEU)


class GPQA_ELLAMIND_DIAMOND_MC_DE(_GPQA_ELLAMIND_DE_Base):
    """German GPQA - MC format, diamond subset (198 hardest items)."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "GPQA_ELLAMIND_DIAMOND_MC_DE"
    _DIAMOND_ONLY = True
    TASK_STYLER = MCStyle().for_language(Language.DEU)


class GPQA_ELLAMIND_DIAMOND_CLOZE_DE(_GPQA_ELLAMIND_DE_Base):
    """German GPQA - Cloze format, diamond subset (198 hardest items)."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "GPQA_ELLAMIND_DIAMOND_CLOZE_DE"
    _DIAMOND_ONLY = True
    TASK_STYLER = ClozeStyle().for_language(Language.DEU)


class GPQA_ELLAMIND_BPB_DE(_GPQA_ELLAMIND_DE_Base):
    """German GPQA - BPB format (all 448 items)."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "GPQA_ELLAMIND_BPB_DE"
    TASK_STYLER = BPBStyle().for_language(Language.DEU)


class GPQA_ELLAMIND_DIAMOND_BPB_DE(_GPQA_ELLAMIND_DE_Base):
    """German GPQA - BPB format, diamond subset (198 hardest items)."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "GPQA_ELLAMIND_DIAMOND_BPB_DE"
    _DIAMOND_ONLY = True
    TASK_STYLER = BPBStyle().for_language(Language.DEU)
