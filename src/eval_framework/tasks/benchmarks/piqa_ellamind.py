"""German PIQA (EllaMind) tasks.

https://huggingface.co/datasets/ellamind/piqa-multilingual

PIQA supplies separate easy and hard distractors.
"""

from typing import Literal

from eval_framework.choices import EllaMindReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.subjects import ListOfSubjects
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import BPBStyle, ClozeStyle, MCStyle, TaskStyler


def piqa_reader(distractor_level: Literal["easy", "hard"]) -> EllaMindReader:
    """Reads PIQA items: a single easy/hard distractor per level, shuffled in with the correct solution."""
    return EllaMindReader(
        distractor_level,
        question_field="goal",
        correct_field="correct_solution",
        easy_field="easy_distractor",
        hard_field="hard_distractor",
        singular=True,
    )


_QUESTION_PREFIX = "Ziel: "
_CUE_TEXT = "Antwort:"

# One styler per format (all sharing the German prefix/cue). The easy/hard distractor axis belongs to
# the reader, so it is orthogonal to the styler choice.
PIQA_ELLAMIND_CLOZE_STYLER = ClozeStyle(question_prefix=_QUESTION_PREFIX, cue_text=_CUE_TEXT)
PIQA_ELLAMIND_MC_STYLER = MCStyle(question_prefix=_QUESTION_PREFIX, cue_text=_CUE_TEXT)
PIQA_ELLAMIND_BPB_STYLER = BPBStyle(question_prefix=_QUESTION_PREFIX, cue_text=_CUE_TEXT)


def _piqa_ellamind_benchmark(id: str, styler: TaskStyler, distractor_level: Literal["easy", "hard"]) -> Benchmark:
    """Dataset: https://huggingface.co/datasets/ellamind/piqa-multilingual"""
    dataset_path = "ellamind/piqa-multilingual"
    return ComposedBenchmark.compose(
        id=id,
        styler=styler,
        reader=piqa_reader(distractor_level),
        sample_split="validation",
        fewshot_split="validation",
        subjects=ListOfSubjects(["deu"]),
        dataset_policy=pinned_by_framework(dataset_path),
        language=Language.DEU,
    )


def piqa_ellamind_benchmarks() -> list[Benchmark]:
    return [
        _piqa_ellamind_benchmark("PIQA_ELLAMIND_CLOZE_EASY_DE", PIQA_ELLAMIND_CLOZE_STYLER, "easy"),
        _piqa_ellamind_benchmark("PIQA_ELLAMIND_CLOZE_HARD_DE", PIQA_ELLAMIND_CLOZE_STYLER, "hard"),
        _piqa_ellamind_benchmark("PIQA_ELLAMIND_MC_EASY_DE", PIQA_ELLAMIND_MC_STYLER, "easy"),
        _piqa_ellamind_benchmark("PIQA_ELLAMIND_MC_HARD_DE", PIQA_ELLAMIND_MC_STYLER, "hard"),
        _piqa_ellamind_benchmark("PIQA_ELLAMIND_BPB_DE", PIQA_ELLAMIND_BPB_STYLER, "easy"),
    ]
