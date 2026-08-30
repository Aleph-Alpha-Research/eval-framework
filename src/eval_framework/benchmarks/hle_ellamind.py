"""German HLE (Humanity's Last Exam, EllaMind) tasks.

https://huggingface.co/datasets/ellamind/hle-multilingual

HLE uses a single distractor set (``incorrect_answers``). The NATIVE variants restrict evaluation to the
items that are natively multiple-choice in the original benchmark (``answer_type == "multipleChoice"``).
"""

from typing import Any, final, override

from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.eval_kind import Choice
from eval_framework.subjects import ListOfSubjects
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy, Subset
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import BPBStyle, ClozeStyle, MCStyle, TaskStyler, shuffle_correct_with_distractors


@final
class HleReader(ChoiceReader):
    """Reads an HLE item: a single ``incorrect_answers`` distractor set, shuffled in with the correct answer."""

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        choices, correct_index = shuffle_correct_with_distractors(
            correct=item["correct_answer"],
            distractors=item["incorrect_answers"],
            seed_text=item["question"] + item["correct_answer"],
        )
        return ChoiceFields(raw_question=item["question"], choices=choices, correct_index=correct_index)


def _hle_ellamind_benchmark(id: str, styler: TaskStyler, dataset: DatasetPolicy | None = None) -> Benchmark:
    kind = Choice(reader=HleReader(), styler=styler)
    dataset_policy = dataset if dataset is not None else pinned_by_framework("ellamind/hle-multilingual")
    return ComposedBenchmark.compose(
        id=id,
        kind=kind,
        sample_split="test",
        fewshot_split="test",
        subjects=ListOfSubjects(["deu"]),
        dataset_policy=dataset_policy,
        language=Language.DEU,
    )


def _hle_ellamind_native_benchmark(id: str, styler: TaskStyler, dataset: DatasetPolicy | None = None) -> Benchmark:
    source = dataset if dataset is not None else pinned_by_framework("ellamind/hle-multilingual")
    return _hle_ellamind_benchmark(id, styler, Subset(source, keep=lambda row: row["answer_type"] == "multipleChoice"))


def hle_ellamind_mc_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _hle_ellamind_benchmark("HLE_ELLAMIND_MC_DE", MCStyle.for_language(Language.DEU), dataset)


def hle_ellamind_cloze_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _hle_ellamind_benchmark("HLE_ELLAMIND_CLOZE_DE", ClozeStyle.for_language(Language.DEU), dataset)


def hle_ellamind_mc_native_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _hle_ellamind_native_benchmark("HLE_ELLAMIND_MC_NATIVE_DE", MCStyle.for_language(Language.DEU), dataset)


def hle_ellamind_cloze_native_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _hle_ellamind_native_benchmark(
        "HLE_ELLAMIND_CLOZE_NATIVE_DE", ClozeStyle.for_language(Language.DEU), dataset
    )


def hle_ellamind_bpb_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _hle_ellamind_benchmark("HLE_ELLAMIND_BPB_DE", BPBStyle.for_language(Language.DEU), dataset)


HLE_ELLAMIND_BENCHMARKS: list[Benchmark] = [
    hle_ellamind_mc_de(),
    hle_ellamind_cloze_de(),
    hle_ellamind_mc_native_de(),
    hle_ellamind_cloze_native_de(),
    hle_ellamind_bpb_de(),
]
