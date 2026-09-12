"""PIQA (physical commonsense QA).

https://huggingface.co/datasets/ybisk/piqa

Each item is a goal with two candidate solutions (``sol1``, ``sol2``); ``label`` selects the correct one.
The base task scores the full solution text (cloze); the OLMES variant shows the solutions as lettered
options (multiple choice); the IDK variant lets the model abstain with an "I do not know" answer.
"""

from typing import Any, final, override

from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.eval_kind import Choice
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_frozen
from eval_framework.tasks.task_style import ClozeStyle, MCStyle, TaskStyler

_IDK_PREAMBLE = (
    "Complete the sentence only if you are confident, since mistakes may be penalised, while correct "
    "answers receive points. It is acceptable to answer with 'I do not know' if you are unsure, and "
    "you will receive 0 points."
)


@final
class PiqaReader(ChoiceReader):
    """Reads a PIQA item: the goal and its two candidate solutions (``sol1``, ``sol2``) in fixed order."""

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        return ChoiceFields(
            raw_question=item["goal"],
            choices=[item["sol1"], item["sol2"]],
            correct_index=item["label"],
        )


def _piqa_benchmark(
    id: str,
    styler: TaskStyler,
    *,
    sample_split: str,
    fewshot_split: str,
    dataset: DatasetPolicy | None = None,
) -> Benchmark:
    kind = Choice(reader=PiqaReader(), styler=styler)
    dataset_policy = dataset if dataset is not None else pinned_frozen("ybisk/piqa")
    return ComposedBenchmark.compose(
        id=id,
        kind=kind,
        sample_split=sample_split,
        fewshot_split=fewshot_split,
        dataset_policy=dataset_policy,
        language=Language.ENG,
    )


def piqa(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _piqa_benchmark("PIQA", ClozeStyle(), sample_split="validation", fewshot_split="test", dataset=dataset)


def piqa_olmes(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _piqa_benchmark(
        "PIQA_OLMES",
        MCStyle(question_prefix="Goal: ", space_prefixed_labels=True),
        sample_split="train",
        fewshot_split="train",
        dataset=dataset,
    )


def piqa_idk(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _piqa_benchmark(
        "PIQA_IDK",
        ClozeStyle(initial_prompt=lambda _subject: _IDK_PREAMBLE).with_abstention_option(" I do not know"),
        sample_split="validation",
        fewshot_split="test",
        dataset=dataset,
    )


PIQA_BENCHMARKS: list[Benchmark] = [piqa(), piqa_olmes(), piqa_idk()]
