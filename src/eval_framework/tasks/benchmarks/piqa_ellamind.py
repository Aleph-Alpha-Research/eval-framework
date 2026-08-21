"""German PIQA (EllaMind) tasks.

https://huggingface.co/datasets/ellamind/piqa-multilingual

PIQA supplies separate easy and hard distractors. Each base class uses a
``_DISTRACTOR_LEVEL`` class attribute (``"easy"`` or ``"hard"``) that the registered
subclass overrides.
"""

from typing import Any, Literal

from eval_framework.composed import ComposedBenchmark, ComposedEval
from eval_framework.contract import Benchmark
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_revisions import HF_REVISIONS_LOCKFILE
from eval_framework.tasks.task_style import BPBStyle, ClozeStyle, MCStyle, shuffle_correct_with_distractors


class _PIQA_ELLAMIND_DE_Base(ComposedEval[str]):
    """Non-registered base for German PIQA (EllaMind) variants.

    Dataset: https://huggingface.co/datasets/ellamind/piqa-multilingual

    Set ``_DISTRACTOR_LEVEL = "easy"`` or ``"hard"`` on the task class, defaults to "easy".
    """

    DATASET_PATH = "ellamind/piqa-multilingual"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "validation"
    SUBJECTS = ["deu"]
    LANGUAGE = Language.DEU
    _DISTRACTOR_LEVEL: Literal["easy", "hard"] = "easy"

    def _shuffled(self, item: dict[str, Any]) -> tuple[list[str], int]:
        """Return (shuffled_solutions, correct_index) for the selected distractor level."""
        distractor = item["easy_distractor"] if self._DISTRACTOR_LEVEL == "easy" else item["hard_distractor"]
        return shuffle_correct_with_distractors(
            correct=item["correct_solution"],
            distractors=[distractor],
            seed_text=item["goal"] + item["correct_solution"],
        )

    def _get_raw_question(self, item: dict[str, Any]) -> str:
        return item["goal"]

    def _get_choices(self, item: dict[str, Any]) -> list[str]:
        return self._shuffled(item)[0]

    def _get_correct_index(self, item: dict[str, Any]) -> int:
        return self._shuffled(item)[1]


class PIQA_ELLAMIND_CLOZE_EASY_DE(_PIQA_ELLAMIND_DE_Base):
    """German PIQA - Cloze format with easy distractor."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "PIQA_ELLAMIND_CLOZE_EASY_DE"
    TASK_STYLER = ClozeStyle(question_prefix="Ziel: ", cue_text="Antwort:")


class PIQA_ELLAMIND_CLOZE_HARD_DE(_PIQA_ELLAMIND_DE_Base):
    """German PIQA - Cloze format with hard distractor."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "PIQA_ELLAMIND_CLOZE_HARD_DE"
    _DISTRACTOR_LEVEL = "hard"
    TASK_STYLER = ClozeStyle(question_prefix="Ziel: ", cue_text="Antwort:")


class PIQA_ELLAMIND_MC_EASY_DE(_PIQA_ELLAMIND_DE_Base):
    """German PIQA - MC format with easy distractor."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "PIQA_ELLAMIND_MC_EASY_DE"
    TASK_STYLER = MCStyle(question_prefix="Ziel: ", cue_text="Antwort:")


class PIQA_ELLAMIND_MC_HARD_DE(_PIQA_ELLAMIND_DE_Base):
    """German PIQA - MC format with hard distractor."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "PIQA_ELLAMIND_MC_HARD_DE"
    _DISTRACTOR_LEVEL = "hard"
    TASK_STYLER = MCStyle(question_prefix="Ziel: ", cue_text="Antwort:")


class PIQA_ELLAMIND_BPB_DE(_PIQA_ELLAMIND_DE_Base):
    """German PIQA - BPB format (distractor set is irrelevant for BPB)."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "PIQA_ELLAMIND_BPB_DE"
    TASK_STYLER = BPBStyle(question_prefix="Ziel: ", cue_text="Antwort:")


def piqa_ellamind_benchmarks() -> list[Benchmark]:
    return [
        ComposedBenchmark.from_base_task(PIQA_ELLAMIND_CLOZE_EASY_DE),
        ComposedBenchmark.from_base_task(PIQA_ELLAMIND_CLOZE_HARD_DE),
        ComposedBenchmark.from_base_task(PIQA_ELLAMIND_MC_EASY_DE),
        ComposedBenchmark.from_base_task(PIQA_ELLAMIND_MC_HARD_DE),
        ComposedBenchmark.from_base_task(PIQA_ELLAMIND_BPB_DE)
    ]
