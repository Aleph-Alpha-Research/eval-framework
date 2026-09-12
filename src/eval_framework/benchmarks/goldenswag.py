"""GoldenSwag: https://huggingface.co/datasets/PleIAs/GoldenSwag

A curated subset of HellaSwag, read identically (sentence completion, scored as cloze — see
``HellaswagReader``). ``GoldenSwag_IDK`` additionally lets the model abstain with an "I do not know" answer.
"""

from eval_framework.benchmarks.hellaswag import HellaswagReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.eval_kind import Choice
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import ClozeStyle, TaskStyler

_IDK_PREAMBLE = (
    "Complete the sentence only if you are confident, since mistakes may be penalised, while correct "
    "completions receive points. It is acceptable to answer with 'I do not know' if you are unsure, "
    "and you will receive 0 points."
)


def _goldenswag_benchmark(id: str, styler: TaskStyler, dataset: DatasetPolicy | None = None) -> Benchmark:
    kind = Choice(reader=HellaswagReader(), styler=styler)
    dataset_policy = dataset if dataset is not None else pinned_by_framework("PleIAs/GoldenSwag")
    return ComposedBenchmark.compose(
        id=id,
        kind=kind,
        sample_split="validation",
        fewshot_split="validation",
        dataset_policy=dataset_policy,
        language=Language.ENG,
    )


def goldenswag(dataset: DatasetPolicy | None = None) -> Benchmark:
    cloze = ClozeStyle(question_prefix="", cue_text="", trailing_newline=False)
    return _goldenswag_benchmark("GoldenSwag", cloze, dataset)


def goldenswag_idk(dataset: DatasetPolicy | None = None) -> Benchmark:
    cloze = ClozeStyle(
        question_prefix="", cue_text="", trailing_newline=False, initial_prompt=lambda _subject: _IDK_PREAMBLE
    )
    styler = cloze.with_abstention_option(" I do not know.")
    return _goldenswag_benchmark("GoldenSwag_IDK", styler, dataset)


GOLDENSWAG_BENCHMARKS: list[Benchmark] = [goldenswag(), goldenswag_idk()]
