"""German Social IQa (EllaMind) tasks.

https://huggingface.co/datasets/ellamind/siqa-multilingual

SIQA supplies separate easy and hard distractors. Each base class uses a
``_DISTRACTOR_LEVEL`` class attribute (``"easy"`` or ``"hard"``) that the registered
subclass overrides.
"""

from typing import Any, Literal

from eval_framework.tasks.base import BaseTask, Language
from eval_framework.tasks.dataset_revisions import HF_REVISIONS_LOCKFILE
from eval_framework.tasks.task_style import BPBStyle, ClozeStyle, MCStyle, shuffle_correct_with_distractors
from eval_framework.tasks.utils import get_n_letters


class _SIQA_ELLAMIND_DE_Base(BaseTask[str]):
    """Non-registered base for German SIQA (EllaMind) variants.

    Dataset: https://huggingface.co/datasets/ellamind/siqa-multilingual
    """

    DATASET_PATH = "ellamind/siqa-multilingual"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "validation"
    SUBJECTS = ["deu"]
    LANGUAGE = Language.DEU
    PERTURBATION_UNMODIFIABLE_WORDS = ["Kontext", "Frage"] + get_n_letters(4)
    _DISTRACTOR_LEVEL: Literal["easy", "hard"] = "easy"

    def _shuffled(self, item: dict[str, Any]) -> tuple[list[str], int]:
        distractors = item["easy_distractors"] if self._DISTRACTOR_LEVEL == "easy" else item["hard_distractors"]
        return shuffle_correct_with_distractors(
            correct=item["correct_answer"],
            distractors=distractors,
            seed_text=item["question"] + item["correct_answer"],
        )

    def _get_raw_question(self, item: dict[str, Any]) -> str:
        return f"{item['context']} {item['question']}"

    def _get_choices(self, item: dict[str, Any]) -> list[str]:
        return self._shuffled(item)[0]

    def _get_correct_index(self, item: dict[str, Any]) -> int:
        return self._shuffled(item)[1]


class SIQA_ELLAMIND_MC_EASY_DE(_SIQA_ELLAMIND_DE_Base):
    """German SIQA - MC format with easy distractors."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "SIQA_ELLAMIND_MC_EASY_DE"
    TASK_STYLER = MCStyle().for_language(Language.DEU)


class SIQA_ELLAMIND_MC_HARD_DE(_SIQA_ELLAMIND_DE_Base):
    """German SIQA - MC format with hard distractors."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "SIQA_ELLAMIND_MC_HARD_DE"
    _DISTRACTOR_LEVEL = "hard"
    TASK_STYLER = MCStyle().for_language(Language.DEU)


class SIQA_ELLAMIND_CLOZE_EASY_DE(_SIQA_ELLAMIND_DE_Base):
    """German SIQA - Cloze format with easy distractors."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "SIQA_ELLAMIND_CLOZE_EASY_DE"
    TASK_STYLER = ClozeStyle().for_language(Language.DEU)


class SIQA_ELLAMIND_CLOZE_HARD_DE(_SIQA_ELLAMIND_DE_Base):
    """German SIQA - Cloze format with hard distractors."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "SIQA_ELLAMIND_CLOZE_HARD_DE"
    _DISTRACTOR_LEVEL = "hard"
    TASK_STYLER = ClozeStyle().for_language(Language.DEU)


class SIQA_ELLAMIND_BPB_DE(SIQA_ELLAMIND_CLOZE_EASY_DE):
    """German SIQA - BPB format (distractor set is irrelevant for BPB)."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "SIQA_ELLAMIND_BPB_DE"
    TASK_STYLER = BPBStyle().for_language(Language.DEU)
