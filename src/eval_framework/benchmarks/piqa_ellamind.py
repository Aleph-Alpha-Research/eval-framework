"""German PIQA (EllaMind) tasks.

https://huggingface.co/datasets/ellamind/piqa-multilingual

PIQA supplies separate easy and hard distractors.
"""

from typing import Any, Literal, final, override

from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.subjects import ListOfSubjects
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import BPBStyle, ClozeStyle, MCStyle, TaskStyler, shuffle_correct_with_distractors


@final
class PiqaReader(ChoiceReader):
    """Reads a PIQA item: a single easy/hard distractor per level, shuffled in with the correct solution."""

    def __init__(self, distractor_level: Literal["easy", "hard"]) -> None:
        self._distractor_level = distractor_level

    @override
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


def _piqa_ellamind_benchmark(
    id: str, styler: TaskStyler, distractor_level: Literal["easy", "hard"], dataset: DatasetPolicy | None
) -> Benchmark:
    return ComposedBenchmark.compose(
        id=id,
        styler=styler,
        reader=PiqaReader(distractor_level),
        sample_split="validation",
        fewshot_split="validation",
        subjects=ListOfSubjects(["deu"]),
        dataset_policy=dataset if dataset is not None else pinned_by_framework("ellamind/piqa-multilingual"),
        language=Language.DEU,
    )


def piqa_ellamind_cloze_easy_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _piqa_ellamind_benchmark(
        "PIQA_ELLAMIND_CLOZE_EASY_DE", ClozeStyle(question_prefix=_QUESTION_PREFIX, cue_text=_CUE_TEXT), "easy", dataset
    )


def piqa_ellamind_cloze_hard_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _piqa_ellamind_benchmark(
        "PIQA_ELLAMIND_CLOZE_HARD_DE", ClozeStyle(question_prefix=_QUESTION_PREFIX, cue_text=_CUE_TEXT), "hard", dataset
    )


def piqa_ellamind_mc_easy_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _piqa_ellamind_benchmark(
        "PIQA_ELLAMIND_MC_EASY_DE", MCStyle(question_prefix=_QUESTION_PREFIX, cue_text=_CUE_TEXT), "easy", dataset
    )


def piqa_ellamind_mc_hard_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _piqa_ellamind_benchmark(
        "PIQA_ELLAMIND_MC_HARD_DE", MCStyle(question_prefix=_QUESTION_PREFIX, cue_text=_CUE_TEXT), "hard", dataset
    )


def piqa_ellamind_bpb_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _piqa_ellamind_benchmark(
        "PIQA_ELLAMIND_BPB_DE", BPBStyle(question_prefix=_QUESTION_PREFIX, cue_text=_CUE_TEXT), "easy", dataset
    )


PIQA_ELLAMIND_BENCHMARKS: list[Benchmark] = [
    piqa_ellamind_cloze_easy_de(),
    piqa_ellamind_cloze_hard_de(),
    piqa_ellamind_mc_easy_de(),
    piqa_ellamind_mc_hard_de(),
    piqa_ellamind_bpb_de(),
]
