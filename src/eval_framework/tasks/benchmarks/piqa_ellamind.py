"""German PIQA (EllaMind) tasks.

https://huggingface.co/datasets/ellamind/piqa-multilingual

PIQA supplies separate easy and hard distractors. Which one a variant uses is a data-access
concern, carried by the ``PiqaReader`` handed to each benchmark rather than by the task class.
"""

from typing import Any, Literal

from eval_framework.composed import ChoiceFields, ChoiceReader, ComposedBenchmark, ComposedEval
from eval_framework.contract import Benchmark
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_revisions import HF_REVISIONS_LOCKFILE
from eval_framework.tasks.task_style import BPBStyle, ClozeStyle, MCStyle, shuffle_correct_with_distractors


class PiqaReader(ChoiceReader):
    """Reads PIQA items into choice fields, shuffling the correct solution in among one distractor.

    PIQA ships separate easy and hard distractors; ``distractor_level`` selects which to use. BPB
    ignores the distractor set, so the level is immaterial there.
    """

    def __init__(self, distractor_level: Literal["easy", "hard"]) -> None:
        self._distractor_level = distractor_level

    def read(self, item: dict[str, Any]) -> ChoiceFields:
        distractor = item["easy_distractor"] if self._distractor_level == "easy" else item["hard_distractor"]
        choices, correct_index = shuffle_correct_with_distractors(
            correct=item["correct_solution"],
            distractors=[distractor],
            seed_text=item["goal"] + item["correct_solution"],
        )
        return ChoiceFields(raw_question=item["goal"], choices=choices, correct_index=correct_index)


class _PIQA_ELLAMIND_DE_Base(ComposedEval[str]):
    """Non-registered base for German PIQA (EllaMind) variants.

    Dataset: https://huggingface.co/datasets/ellamind/piqa-multilingual
    """

    DATASET_PATH = "ellamind/piqa-multilingual"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "validation"
    SUBJECTS = ["deu"]
    LANGUAGE = Language.DEU


class PIQA_ELLAMIND_CLOZE_EASY_DE(_PIQA_ELLAMIND_DE_Base):
    """German PIQA - Cloze format with easy distractor."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "PIQA_ELLAMIND_CLOZE_EASY_DE"
    TASK_STYLER = ClozeStyle(question_prefix="Ziel: ", cue_text="Antwort:")


class PIQA_ELLAMIND_CLOZE_HARD_DE(_PIQA_ELLAMIND_DE_Base):
    """German PIQA - Cloze format with hard distractor."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "PIQA_ELLAMIND_CLOZE_HARD_DE"
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
    TASK_STYLER = MCStyle(question_prefix="Ziel: ", cue_text="Antwort:")


class PIQA_ELLAMIND_BPB_DE(_PIQA_ELLAMIND_DE_Base):
    """German PIQA - BPB format (distractor set is irrelevant for BPB)."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "PIQA_ELLAMIND_BPB_DE"
    TASK_STYLER = BPBStyle(question_prefix="Ziel: ", cue_text="Antwort:")


def _piqa_benchmark(task: type[_PIQA_ELLAMIND_DE_Base], distractor_level: Literal["easy", "hard"]) -> Benchmark:
    return ComposedBenchmark.from_base(task, PiqaReader(distractor_level))


def piqa_ellamind_benchmarks() -> list[Benchmark]:
    return [
        _piqa_benchmark(PIQA_ELLAMIND_CLOZE_EASY_DE, "easy"),
        _piqa_benchmark(PIQA_ELLAMIND_CLOZE_HARD_DE, "hard"),
        _piqa_benchmark(PIQA_ELLAMIND_MC_EASY_DE, "easy"),
        _piqa_benchmark(PIQA_ELLAMIND_MC_HARD_DE, "hard"),
        _piqa_benchmark(PIQA_ELLAMIND_BPB_DE, "easy"),
    ]
