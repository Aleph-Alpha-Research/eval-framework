"""German SimpleQA (verified, EllaMind) tasks.

https://huggingface.co/datasets/ellamind/simpleqa-verified-multilingual

SimpleQA supplies separate easy and hard distractors. The ``answer_aliases`` field is unused.
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
class SimpleqaReader(ChoiceReader):
    """Reads a SimpleQA item: easy/hard distractor lists per level, shuffled in with the correct answer."""

    def __init__(self, distractor_level: Literal["easy", "hard"]) -> None:
        self._distractor_level = distractor_level

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        distractors = item["easy_distractors"] if self._distractor_level == "easy" else item["hard_distractors"]
        choices, correct_index = shuffle_correct_with_distractors(
            correct=item["answer"],
            distractors=distractors,
            seed_text=item["question"] + item["answer"],
        )
        return ChoiceFields(raw_question=item["question"], choices=choices, correct_index=correct_index)


def _simpleqa_ellamind_benchmark(
    id: str, styler: TaskStyler, distractor_level: Literal["easy", "hard"], dataset: DatasetPolicy | None
) -> Benchmark:
    return ComposedBenchmark.compose(
        id=id,
        styler=styler,
        reader=SimpleqaReader(distractor_level),
        sample_split="eval",
        fewshot_split="eval",
        subjects=ListOfSubjects(["deu"]),
        dataset_policy=dataset
        if dataset is not None
        else pinned_by_framework("ellamind/simpleqa-verified-multilingual"),
        language=Language.DEU,
    )


def simpleqa_ellamind_mc_easy_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _simpleqa_ellamind_benchmark(
        "SIMPLEQA_ELLAMIND_MC_EASY_DE", MCStyle.for_language(Language.DEU), "easy", dataset
    )


def simpleqa_ellamind_mc_hard_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _simpleqa_ellamind_benchmark(
        "SIMPLEQA_ELLAMIND_MC_HARD_DE", MCStyle.for_language(Language.DEU), "hard", dataset
    )


def simpleqa_ellamind_cloze_easy_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _simpleqa_ellamind_benchmark(
        "SIMPLEQA_ELLAMIND_CLOZE_EASY_DE", ClozeStyle.for_language(Language.DEU), "easy", dataset
    )


def simpleqa_ellamind_cloze_hard_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _simpleqa_ellamind_benchmark(
        "SIMPLEQA_ELLAMIND_CLOZE_HARD_DE", ClozeStyle.for_language(Language.DEU), "hard", dataset
    )


def simpleqa_ellamind_bpb_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _simpleqa_ellamind_benchmark(
        "SIMPLEQA_ELLAMIND_BPB_DE", BPBStyle.for_language(Language.DEU), "easy", dataset
    )


SIMPLEQA_ELLAMIND_BENCHMARKS: list[Benchmark] = [
    simpleqa_ellamind_mc_easy_de(),
    simpleqa_ellamind_mc_hard_de(),
    simpleqa_ellamind_cloze_easy_de(),
    simpleqa_ellamind_cloze_hard_de(),
    simpleqa_ellamind_bpb_de(),
]
