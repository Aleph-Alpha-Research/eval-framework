"""German GPQA (Graduate-level Professional QA, EllaMind) tasks.

https://huggingface.co/datasets/ellamind/gpqa-multilingual

GPQA uses a single distractor set (``incorrect_answers``). The diamond variants restrict evaluation to
the diamond subset — the 198 hardest questions (``is_diamond``) from the original GPQA-Diamond benchmark.
"""

from typing import Any, final, override

from datasets import DatasetDict

from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.subjects import ListOfSubjects
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetLoader, DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import BPBStyle, ClozeStyle, MCStyle, TaskStyler, shuffle_correct_with_distractors


@final
class GpqaReader(ChoiceReader):
    """Reads a GPQA item: a single ``incorrect_answers`` distractor set, shuffled in with the correct answer."""

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        choices, correct_index = shuffle_correct_with_distractors(
            correct=item["correct_answer"],
            distractors=item["incorrect_answers"],
            seed_text=item["question"] + item["correct_answer"],
        )
        return ChoiceFields(raw_question=item["question"], choices=choices, correct_index=correct_index)


@final
class _DiamondFilteredLoader(DatasetLoader):
    """Restricts a loader's every split to the diamond subset (``is_diamond``)."""

    def __init__(self, inner: DatasetLoader) -> None:
        self._inner = inner

    @override
    def load(self, name: str | None) -> DatasetDict:
        loaded = self._inner.load(name)
        return DatasetDict({split: data.filter(lambda row: row["is_diamond"]) for split, data in loaded.items()})

    @override
    def metadata(self) -> dict[str, str]:
        return self._inner.metadata()


@final
class _DiamondOnly(DatasetPolicy):
    """Wraps a dataset policy to serve only the diamond subset — the 198 hardest GPQA questions."""

    def __init__(self, inner: DatasetPolicy) -> None:
        self._inner = inner

    @override
    def loader(self, custom_hf_revision: str | None) -> DatasetLoader:
        return _DiamondFilteredLoader(self._inner.loader(custom_hf_revision))

    @override
    def documentation(self) -> str:
        return self._inner.documentation()


def _gpqa_ellamind_benchmark(id: str, styler: TaskStyler, dataset: DatasetPolicy | None) -> Benchmark:
    return ComposedBenchmark.compose(
        id=id,
        styler=styler,
        reader=GpqaReader(),
        sample_split="train",
        fewshot_split="train",
        subjects=ListOfSubjects(["deu"]),
        dataset_policy=dataset if dataset is not None else pinned_by_framework("ellamind/gpqa-multilingual"),
        language=Language.DEU,
    )


def _gpqa_ellamind_diamond_benchmark(id: str, styler: TaskStyler, dataset: DatasetPolicy | None) -> Benchmark:
    source = dataset if dataset is not None else pinned_by_framework("ellamind/gpqa-multilingual")
    return _gpqa_ellamind_benchmark(id, styler, _DiamondOnly(source))


def gpqa_ellamind_mc_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _gpqa_ellamind_benchmark("GPQA_ELLAMIND_MC_DE", MCStyle.for_language(Language.DEU), dataset)


def gpqa_ellamind_cloze_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _gpqa_ellamind_benchmark("GPQA_ELLAMIND_CLOZE_DE", ClozeStyle.for_language(Language.DEU), dataset)


def gpqa_ellamind_bpb_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _gpqa_ellamind_benchmark("GPQA_ELLAMIND_BPB_DE", BPBStyle.for_language(Language.DEU), dataset)


def gpqa_ellamind_diamond_mc_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _gpqa_ellamind_diamond_benchmark("GPQA_ELLAMIND_DIAMOND_MC_DE", MCStyle.for_language(Language.DEU), dataset)


def gpqa_ellamind_diamond_cloze_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _gpqa_ellamind_diamond_benchmark(
        "GPQA_ELLAMIND_DIAMOND_CLOZE_DE", ClozeStyle.for_language(Language.DEU), dataset
    )


def gpqa_ellamind_diamond_bpb_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _gpqa_ellamind_diamond_benchmark(
        "GPQA_ELLAMIND_DIAMOND_BPB_DE", BPBStyle.for_language(Language.DEU), dataset
    )


GPQA_ELLAMIND_BENCHMARKS: list[Benchmark] = [
    gpqa_ellamind_mc_de(),
    gpqa_ellamind_cloze_de(),
    gpqa_ellamind_diamond_mc_de(),
    gpqa_ellamind_diamond_cloze_de(),
    gpqa_ellamind_bpb_de(),
    gpqa_ellamind_diamond_bpb_de(),
]
