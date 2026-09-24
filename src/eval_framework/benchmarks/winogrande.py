"""Winogrande: https://huggingface.co/datasets/allenai/winogrande

Pronoun-resolution sentences with a blank ``_`` filled by ``option1`` or ``option2``; ``answer`` selects
the correct one. The registered task uses partial evaluation: each item becomes two samples that score the
shared sentence suffix under each option-augmented prefix — ``p(suffix | prefix + option)``. ``WinograndeReader``
and ``PartialEval`` live here and are reused by the multilingual EllaMind variants.
"""

from typing import TYPE_CHECKING, Any, final, override

from eval_framework.answer import PickFromCandidates
from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.eval_kind import EvalKind, SampleBody
from eval_framework.fewshot import ChoiceRenderer, FewShot, SampleSplit
from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import PartialEvalAccuracy
from eval_framework.subjects import ListOfSubjects
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import ClozeStyle

if TYPE_CHECKING:
    from eval_framework.metrics.base import BaseMetric

_WINOGRANDE_ANSWER_STR_TO_NUM = {"1": 0, "2": 1}


@final
class WinograndeReader(ChoiceReader):
    """Reads a Winogrande item: the shown question is the sentence prefix (before the blank ``_``); each
    choice is an option completed by the shared suffix (after the blank)."""

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        prefix, suffix = item["sentence"].split("_", 1)
        prefix = prefix.replace("  ", " ").strip()
        suffix = suffix.replace("  ", " ")
        return ChoiceFields(
            raw_question=prefix,
            choices=[item["option1"] + suffix, item["option2"] + suffix],
            correct_index=_WINOGRANDE_ANSWER_STR_TO_NUM[item["answer"]],
        )


@final
class PartialEval(EvalKind):
    """Winogrande partial evaluation: one item becomes two samples, each scoring the shared sentence
    suffix under one option — ``p(suffix | prefix + option)``. ``PartialEvalAccuracy`` pairs the two
    (consecutive ids) and picks the option under which the suffix is likelier."""

    def __init__(self) -> None:
        self._reader = WinograndeReader()

    @override
    def metrics(self) -> list[type["BaseMetric"]]:
        return [PartialEvalAccuracy]

    @override
    def samples(self, item: dict[str, Any]) -> list[SampleBody]:
        prefix, suffix = item["sentence"].split("_", 1)
        prefix = prefix.replace("  ", " ")  # keep the trailing space so "prefix + option" is well-formed
        suffix = suffix.replace("  ", " ")
        correct_index = _WINOGRANDE_ANSWER_STR_TO_NUM[item["answer"]]
        return [
            SampleBody(
                prompt=f"{prefix}{option}",
                cue="",  # partial evaluation scores the suffix directly, with no assistant cue
                possible_completions=[suffix],
                ground_truth=str(opt_index == correct_index),
            )
            for opt_index, option in enumerate([item["option1"], item["option2"]])
        ]


def winogrande_cloze(dataset: DatasetPolicy | None = None) -> Benchmark:
    # "Cloze" is the registered name, but the task is partial evaluation; its few-shot demonstrations
    # render as ordinary cloze (the prefix, then the correct option + suffix).
    fewshot_styler = ClozeStyle(question_prefix="", trailing_newline=False, cue_text="")
    dataset_policy = dataset if dataset is not None else pinned_by_framework("allenai/winogrande")
    return ComposedBenchmark.compose(
        id="WINOGRANDECloze",
        display_name="WinograndeCloze",
        kind=PartialEval(),
        answer=PickFromCandidates(),
        sample_split="train",
        fewshot=FewShot(SampleSplit(), ChoiceRenderer(WinograndeReader(), fewshot_styler)),
        subjects=ListOfSubjects(["winogrande_xl"]),
        dataset_policy=dataset_policy,
        language=Language.ENG,
    )


WINOGRANDE_BENCHMARKS: list[Benchmark] = [winogrande_cloze()]
