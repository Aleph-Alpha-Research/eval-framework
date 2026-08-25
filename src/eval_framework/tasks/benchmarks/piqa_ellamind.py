"""German PIQA (EllaMind) tasks.

https://huggingface.co/datasets/ellamind/piqa-multilingual

PIQA supplies separate easy and hard distractors. Which one a variant uses is a data-access
concern, carried by the ``PiqaReader`` handed to each benchmark rather than by the task class.
"""

from typing import Any, Literal

from eval_framework.composed import ChoiceFields, ChoiceReader, ComposedBenchmark, ComposedEval
from eval_framework.contract import Benchmark
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_revisions import pinned_by_framework
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


_QUESTION_PREFIX = "Ziel: "
_CUE_TEXT = "Antwort:"

# One styler per format (all sharing the German prefix/cue). The easy/hard distractor axis belongs to
# the reader, so it is orthogonal to the styler choice.
PIQA_ELLAMIND_CLOZE_STYLER = ClozeStyle(question_prefix=_QUESTION_PREFIX, cue_text=_CUE_TEXT)
PIQA_ELLAMIND_MC_STYLER = MCStyle(question_prefix=_QUESTION_PREFIX, cue_text=_CUE_TEXT)
PIQA_ELLAMIND_BPB_STYLER = BPBStyle(question_prefix=_QUESTION_PREFIX, cue_text=_CUE_TEXT)


class PIQA_ELLAMIND_CLOZE_EASY_DE(ComposedEval[str]):
    """German PIQA - Cloze format with easy distractor."""

    NAME = "PIQA_ELLAMIND_CLOZE_EASY_DE"
    TASK_STYLER = PIQA_ELLAMIND_CLOZE_STYLER


class PIQA_ELLAMIND_CLOZE_HARD_DE(ComposedEval[str]):
    """German PIQA - Cloze format with hard distractor."""

    NAME = "PIQA_ELLAMIND_CLOZE_HARD_DE"
    TASK_STYLER = PIQA_ELLAMIND_CLOZE_STYLER


class PIQA_ELLAMIND_MC_EASY_DE(ComposedEval[str]):
    """German PIQA - MC format with easy distractor."""

    NAME = "PIQA_ELLAMIND_MC_EASY_DE"
    TASK_STYLER = PIQA_ELLAMIND_MC_STYLER


class PIQA_ELLAMIND_MC_HARD_DE(ComposedEval[str]):
    """German PIQA - MC format with hard distractor."""

    NAME = "PIQA_ELLAMIND_MC_HARD_DE"
    TASK_STYLER = PIQA_ELLAMIND_MC_STYLER


class PIQA_ELLAMIND_BPB_DE(ComposedEval[str]):
    """German PIQA - BPB format (distractor set is irrelevant for BPB)."""

    NAME = "PIQA_ELLAMIND_BPB_DE"
    TASK_STYLER = PIQA_ELLAMIND_BPB_STYLER


def _piqa_ellamind_benchmark(task: type[ComposedEval[str]], distractor_level: Literal["easy", "hard"]) -> Benchmark:
    """Dataset: https://huggingface.co/datasets/ellamind/piqa-multilingual"""
    dataset_path = "ellamind/piqa-multilingual"
    return ComposedBenchmark.from_base(
        task,
        PiqaReader(distractor_level),
        sample_split="validation",
        fewshot_split="validation",
        subjects=["deu"],
        dataset_policy=pinned_by_framework(dataset_path),
        language=Language.DEU,
    )


def piqa_ellamind_benchmarks() -> list[Benchmark]:
    return [
        _piqa_ellamind_benchmark(PIQA_ELLAMIND_CLOZE_EASY_DE, "easy"),
        _piqa_ellamind_benchmark(PIQA_ELLAMIND_CLOZE_HARD_DE, "hard"),
        _piqa_ellamind_benchmark(PIQA_ELLAMIND_MC_EASY_DE, "easy"),
        _piqa_ellamind_benchmark(PIQA_ELLAMIND_MC_HARD_DE, "hard"),
        _piqa_ellamind_benchmark(PIQA_ELLAMIND_BPB_DE, "easy"),
    ]
