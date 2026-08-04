"""German CommonsenseQA (EllaMind) tasks.

https://huggingface.co/datasets/ellamind/csqa-multilingual

CSQA supplies separate easy and hard distractors. Each base class uses a
``_DISTRACTOR_LEVEL`` class attribute (``"easy"`` or ``"hard"``) that the registered
subclass overrides.
"""

from typing import Any, Literal

from eval_framework.tasks.base import BaseTask, Language
from eval_framework.tasks.dataset_revisions import HF_REVISIONS_LOCKFILE
from eval_framework.tasks.task_style import BPBStyle, ClozeStyle, MCStyle, shuffle_correct_with_distractors
from eval_framework.tasks.utils import get_n_letters


class _CSQA_ELLAMIND_DE_Base(BaseTask[str]):
    """Non-registered base for German CSQA (EllaMind) variants.

    Dataset: https://huggingface.co/datasets/ellamind/csqa-multilingual

    Set ``_DISTRACTOR_LEVEL = "easy"`` or ``"hard"`` on the task class, defaults to "easy".
    """

    DATASET_PATH = "ellamind/csqa-multilingual"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "validation"
    SUBJECTS = ["deu"]
    LANGUAGE = Language.DEU
    PERTURBATION_UNMODIFIABLE_WORDS = ["Frage", "Answer"] + get_n_letters(5)
    _DISTRACTOR_LEVEL: Literal["easy", "hard"] = "easy"

    def _shuffled(self, item: dict[str, Any]) -> tuple[list[str], int]:
        """Return (shuffled_choices, correct_index) for the selected distractor level."""
        distractors = item["easy_distractors"] if self._DISTRACTOR_LEVEL == "easy" else item["hard_distractors"]
        return shuffle_correct_with_distractors(
            correct=item["correct_answer"],
            distractors=distractors,
            seed_text=item["question"] + item["correct_answer"],
        )

    def _get_raw_question(self, item: dict[str, Any]) -> str:
        return item["question"]

    def _get_choices(self, item: dict[str, Any]) -> list[str]:
        return self._shuffled(item)[0]

    def _get_correct_index(self, item: dict[str, Any]) -> int:
        return self._shuffled(item)[1]


class CSQA_ELLAMIND_MC_EASY_DE(_CSQA_ELLAMIND_DE_Base):
    """German CSQA - MC format with easy distractors."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "CSQA_ELLAMIND_MC_EASY_DE"
    TASK_STYLER = MCStyle().for_language(Language.DEU)


class CSQA_ELLAMIND_MC_HARD_DE(_CSQA_ELLAMIND_DE_Base):
    """German CSQA - MC format with hard distractors."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "CSQA_ELLAMIND_MC_HARD_DE"
    _DISTRACTOR_LEVEL = "hard"
    TASK_STYLER = MCStyle().for_language(Language.DEU)


class CSQA_ELLAMIND_CLOZE_EASY_DE(_CSQA_ELLAMIND_DE_Base):
    """German CSQA - Cloze format with easy distractors."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "CSQA_ELLAMIND_CLOZE_EASY_DE"
    TASK_STYLER = ClozeStyle().for_language(Language.DEU)


class CSQA_ELLAMIND_CLOZE_HARD_DE(_CSQA_ELLAMIND_DE_Base):
    """German CSQA - Cloze format with hard distractors."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "CSQA_ELLAMIND_CLOZE_HARD_DE"
    _DISTRACTOR_LEVEL = "hard"
    TASK_STYLER = ClozeStyle().for_language(Language.DEU)


class CSQA_ELLAMIND_BPB_DE(CSQA_ELLAMIND_CLOZE_EASY_DE):
    """German CSQA - BPB format (easy distractors; distractor set is irrelevant for BPB)."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "CSQA_ELLAMIND_BPB_DE"
    TASK_STYLER = BPBStyle().for_language(Language.DEU)
