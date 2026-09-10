"""HellaSwag: https://huggingface.co/datasets/Rowan/hellaswag

Sentence completion: each item gives an activity label and a context, and the model scores which ending
best continues it. Scored as cloze — the prompt is ``"{activity}: {context}"`` and the candidates are the
full endings. ``HellaSwag`` evaluates on the validation split; ``HellaSwag_OLMES`` on the (larger) train
split, matching OLMES.
"""

import re
from typing import Any, final, override

from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.eval_kind import Choice
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import ClozeStyle


def _preprocess(text: str) -> str:
    # HellaSwag text carries wikiHow markup ("[header]", "[title]", ...); reduce it to plain prose.
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


@final
class HellaswagReader(ChoiceReader):
    """Reads a HellaSwag item: the shown text is ``"{activity}: {context}"`` (context is ``ctx_a`` plus a
    capitalised ``ctx_b``), and each choice is a candidate ending. Markup is stripped from all shown text."""

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        activity = _preprocess(item["activity_label"])
        context = _preprocess(item["ctx_a"] + " " + item["ctx_b"].capitalize()).strip()
        return ChoiceFields(
            raw_question=f"{activity}: {context}",
            choices=[_preprocess(ending) for ending in item["endings"]],
            correct_index=int(item["label"]) if item["label"] != "" else 0,
        )


def _hellaswag_benchmark(
    id: str, *, sample_split: str, fewshot_split: str, dataset: DatasetPolicy | None = None
) -> Benchmark:
    kind = Choice(reader=HellaswagReader(), styler=ClozeStyle(question_prefix="", cue_text="", trailing_newline=False))
    dataset_policy = dataset if dataset is not None else pinned_by_framework("Rowan/hellaswag")
    return ComposedBenchmark.compose(
        id=id,
        kind=kind,
        sample_split=sample_split,
        fewshot_split=fewshot_split,
        dataset_policy=dataset_policy,
        language=Language.ENG,
    )


def hellaswag(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _hellaswag_benchmark("HellaSwag", sample_split="validation", fewshot_split="train", dataset=dataset)


def hellaswag_olmes(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _hellaswag_benchmark("HellaSwag_OLMES", sample_split="train", fewshot_split="train", dataset=dataset)


HELLASWAG_BENCHMARKS: list[Benchmark] = [hellaswag(), hellaswag_olmes()]
