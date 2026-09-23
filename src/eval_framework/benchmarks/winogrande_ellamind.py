"""German Winogrande (EllaMind) tasks.

https://huggingface.co/datasets/ellamind/winogrande-multilingual

The sentence contains a blank ``_`` filled by ``option1`` or ``option2``; ``answer`` selects the correct
one. Cloze and MC score the two full "option + suffix" completions; partial evaluation instead scores the
shared suffix under each option-augmented prefix.
"""

from eval_framework.answer import PickFromCandidates
from eval_framework.benchmarks.winogrande import PartialEval, WinograndeReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.fewshot import ChoiceRenderer, SampledFewShot
from eval_framework.subjects import ListOfSubjects
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import ClozeStyle, MCStyle, TaskStyler


def _winogrande_dataset(dataset: DatasetPolicy | None) -> DatasetPolicy:
    return dataset if dataset is not None else pinned_by_framework("ellamind/winogrande-multilingual")


def _winogrande_choice(id: str, styler: TaskStyler, dataset: DatasetPolicy | None = None) -> Benchmark:
    return ComposedBenchmark.choice(
        id=id,
        reader=WinograndeReader(),
        styler=styler,
        sample_split="validation",
        fewshot_split="validation",
        subjects=ListOfSubjects(["deu"]),
        dataset_policy=_winogrande_dataset(dataset),
        language=Language.DEU,
    )


def winogrande_ellamind_cloze_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    styler = ClozeStyle(question_prefix="", trailing_newline=False, cue_text="")
    return _winogrande_choice("WINOGRANDE_ELLAMIND_CLOZE_DE", styler, dataset)


def winogrande_ellamind_mc_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _winogrande_choice("WINOGRANDE_ELLAMIND_MC_DE", MCStyle.for_language(Language.DEU), dataset)


def winogrande_ellamind_partial_eval_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    # Partial evaluation scores the suffix directly; its few-shot demonstrations render as ordinary cloze.
    fewshot_styler = ClozeStyle(question_prefix="", trailing_newline=False, cue_text="")
    return ComposedBenchmark.compose(
        id="WINOGRANDE_ELLAMIND_PARTIAL_EVAL_DE",
        kind=PartialEval(),
        answer=PickFromCandidates(),
        sample_split="validation",
        fewshot=SampledFewShot("validation", ChoiceRenderer(WinograndeReader(), fewshot_styler)),
        subjects=ListOfSubjects(["deu"]),
        dataset_policy=_winogrande_dataset(dataset),
        language=Language.DEU,
    )


WINOGRANDE_ELLAMIND_BENCHMARKS: list[Benchmark] = [
    winogrande_ellamind_cloze_de(),
    winogrande_ellamind_mc_de(),
    winogrande_ellamind_partial_eval_de(),
]
