"""German ARC (EllaMind): https://huggingface.co/datasets/ellamind/arc-multilingual

Grade-school science questions in German. Every slice loads the single German config (``deu``); the
ARC-Easy and ARC-Challenge subjects are the rows of that config tagged by the ``arc_config`` column. Cloze
scores the full answer text, MC the letter labels, BPB only the ground-truth answer's bits-per-byte.
"""

from typing import Any, final, override

from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.subjects import ListOfSubjects
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import BPBStyle, ClozeStyle, MCStyle, TaskStyler, answer_key_to_index


@final
class ArcEllamindReader(ChoiceReader):
    """Reads a German ARC item: the question and its answer options, with the correct one at ``answer_key``
    (a letter A–E or a 1-based number)."""

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        return ChoiceFields(
            raw_question=item["question"],
            choices=item["choices"],
            correct_index=answer_key_to_index(item["answer_key"]),
        )


def _arc_ellamind_benchmark(id: str, styler: TaskStyler, dataset: DatasetPolicy | None = None) -> Benchmark:
    # ARC-Easy and ARC-Challenge share the single "deu" config, tagged by the ``arc_config`` column.
    source = (
        dataset
        if dataset is not None
        else pinned_by_framework("ellamind/arc-multilingual").subject_encoded_in_column(
            config="deu", column="arc_config"
        )
    )
    return ComposedBenchmark.choice(
        id=id,
        reader=ArcEllamindReader(),
        styler=styler,
        sample_split="test",
        fewshot_split="test",
        subjects=ListOfSubjects(["ARC-Easy", "ARC-Challenge"]),
        dataset_policy=source,
        language=Language.DEU,
    )


def arc_ellamind_cloze_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _arc_ellamind_benchmark("ARC_ELLAMIND_CLOZE_DE", ClozeStyle.for_language(Language.DEU), dataset)


def arc_ellamind_mc_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _arc_ellamind_benchmark("ARC_ELLAMIND_MC_DE", MCStyle.for_language(Language.DEU), dataset)


def arc_ellamind_bpb_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _arc_ellamind_benchmark("ARC_ELLAMIND_BPB_DE", BPBStyle.for_language(Language.DEU), dataset)


ARC_ELLAMIND_BENCHMARKS: list[Benchmark] = [
    arc_ellamind_cloze_de(),
    arc_ellamind_mc_de(),
    arc_ellamind_bpb_de(),
]
