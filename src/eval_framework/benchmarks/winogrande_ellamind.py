"""German Winogrande (EllaMind) tasks.

https://huggingface.co/datasets/ellamind/winogrande-multilingual

The sentence contains a blank ``_`` filled by ``option1`` or ``option2``; ``answer`` selects the correct
one. Cloze and MC score the two full "option + suffix" completions; partial evaluation instead scores the
shared suffix under each option-augmented prefix.
"""

from typing import TYPE_CHECKING, Any, final, override

from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark, ResponseType
from eval_framework.eval_kind import Choice, EvalKind, FewshotExample, SampleBody
from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import PartialEvalAccuracy
from eval_framework.subjects import ListOfSubjects
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import ClozeStyle, MCStyle

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
    (consecutive ids) and picks the option under which the suffix is likelier. Few-shot examples render
    as ordinary cloze (the prefix, then the correct option + suffix)."""

    response_type = ResponseType.LOGLIKELIHOODS
    metrics: list[type["BaseMetric"]] = [PartialEvalAccuracy]

    def __init__(self) -> None:
        self._reader = WinograndeReader()
        self._fewshot_styler = ClozeStyle(question_prefix="", trailing_newline=False, cue_text="")

    @override
    def fewshot(self, item: dict[str, Any]) -> FewshotExample:
        fields = self._reader.read(item)
        return FewshotExample(
            prompt=self._fewshot_styler.get_instruction_text(fields.raw_question, fields.choices),
            answer=self._fewshot_styler.get_fewshot_target_text(fields.choices, fields.correct_index),
        )

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


def _winogrande_ellamind_benchmark(id: str, kind: EvalKind, dataset: DatasetPolicy | None = None) -> Benchmark:
    dataset_policy = dataset if dataset is not None else pinned_by_framework("ellamind/winogrande-multilingual")
    return ComposedBenchmark.compose(
        id=id,
        kind=kind,
        sample_split="validation",
        fewshot_split="validation",
        subjects=ListOfSubjects(["deu"]),
        dataset_policy=dataset_policy,
        language=Language.DEU,
    )


def winogrande_ellamind_cloze_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    styler = ClozeStyle(question_prefix="", trailing_newline=False, cue_text="")
    return _winogrande_ellamind_benchmark("WINOGRANDE_ELLAMIND_CLOZE_DE", Choice(WinograndeReader(), styler), dataset)


def winogrande_ellamind_mc_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    styler = MCStyle.for_language(Language.DEU)
    return _winogrande_ellamind_benchmark("WINOGRANDE_ELLAMIND_MC_DE", Choice(WinograndeReader(), styler), dataset)


def winogrande_ellamind_partial_eval_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _winogrande_ellamind_benchmark("WINOGRANDE_ELLAMIND_PARTIAL_EVAL_DE", PartialEval(), dataset)


WINOGRANDE_ELLAMIND_BENCHMARKS: list[Benchmark] = [
    winogrande_ellamind_cloze_de(),
    winogrande_ellamind_mc_de(),
    winogrande_ellamind_partial_eval_de(),
]
