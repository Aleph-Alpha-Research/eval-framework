"""German CommonsenseQA (EllaMind) tasks.

https://huggingface.co/datasets/ellamind/csqa-multilingual

CSQA supplies separate easy and hard distractors.
"""

from typing import Any, Literal, final, override

from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.eval_kind import Choice
from eval_framework.subjects import ListOfSubjects
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import BPBStyle, ClozeStyle, MCStyle, TaskStyler, shuffle_correct_with_distractors


@final
class CsqaReader(ChoiceReader):
    """Reads a CSQA item: easy/hard distractor lists per level, shuffled in with the correct answer."""

    def __init__(self, distractor_level: Literal["easy", "hard"]) -> None:
        self._distractor_level = distractor_level

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        distractors = item["easy_distractors"] if self._distractor_level == "easy" else item["hard_distractors"]
        choices, correct_index = shuffle_correct_with_distractors(
            correct=item["correct_answer"],
            distractors=distractors,
            seed_text=item["question"] + item["correct_answer"],
        )
        return ChoiceFields(raw_question=item["question"], choices=choices, correct_index=correct_index)


def _csqa_ellamind_benchmark(
    id: str, styler: TaskStyler, distractor_level: Literal["easy", "hard"], dataset: DatasetPolicy | None = None
) -> Benchmark:
    kind = Choice(reader=CsqaReader(distractor_level), styler=styler)
    dataset_policy = dataset if dataset is not None else pinned_by_framework("ellamind/csqa-multilingual")
    return ComposedBenchmark.compose(
        id=id,
        kind=kind,
        sample_split="validation",
        fewshot_split="validation",
        subjects=ListOfSubjects(["deu"]),
        dataset_policy=dataset_policy,
        language=Language.DEU,
    )


def csqa_ellamind_mc_easy_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _csqa_ellamind_benchmark("CSQA_ELLAMIND_MC_EASY_DE", MCStyle.for_language(Language.DEU), "easy", dataset)


def csqa_ellamind_mc_hard_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _csqa_ellamind_benchmark("CSQA_ELLAMIND_MC_HARD_DE", MCStyle.for_language(Language.DEU), "hard", dataset)


def csqa_ellamind_cloze_easy_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _csqa_ellamind_benchmark(
        "CSQA_ELLAMIND_CLOZE_EASY_DE", ClozeStyle.for_language(Language.DEU), "easy", dataset
    )


def csqa_ellamind_cloze_hard_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _csqa_ellamind_benchmark(
        "CSQA_ELLAMIND_CLOZE_HARD_DE", ClozeStyle.for_language(Language.DEU), "hard", dataset
    )


def csqa_ellamind_bpb_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _csqa_ellamind_benchmark("CSQA_ELLAMIND_BPB_DE", BPBStyle.for_language(Language.DEU), "easy", dataset)


CSQA_ELLAMIND_BENCHMARKS: list[Benchmark] = [
    csqa_ellamind_mc_easy_de(),
    csqa_ellamind_mc_hard_de(),
    csqa_ellamind_cloze_easy_de(),
    csqa_ellamind_cloze_hard_de(),
    csqa_ellamind_bpb_de(),
]
