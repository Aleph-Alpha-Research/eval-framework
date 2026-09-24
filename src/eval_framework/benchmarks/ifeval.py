"""IFEval: Instruction Following Eval (https://arxiv.org/pdf/2311.07911).

The model follows a natural-language prompt carrying verifiable constraints (word counts, formats, casing, …).
The instruction checks run from a per-sample ``IFEvalMetricContext``, so the task has no gold answer and is
0-shot only.
"""

from typing import TYPE_CHECKING, Any

from eval_framework.answer import ExtractFromCompletion
from eval_framework.composed import ComposedBenchmark, LanguageSpec
from eval_framework.contract import Benchmark
from eval_framework.eval_kind import Generative
from eval_framework.fewshot import NoFewShot
from eval_framework.metrics.completion.ifeval import IFEvalMetric, IFEvalMetricContext
from eval_framework.metrics.completion.language_checker import LanguageRawConsistencyChecker
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework

if TYPE_CHECKING:
    from eval_framework.metrics.base import BaseMetric

IFEVAL_DATASET_PATH = "google/IFEval"
IFEVAL_DE_DATASET_PATH = "jzhang86/de_ifeval"


def _ifeval_context(item: dict[str, Any]) -> IFEvalMetricContext:
    new_kwargs = []
    for d in item["kwargs"]:
        # Some dataset variants type integer kwargs as float; int() below must not truncate anything.
        assert all(v.is_integer() for v in d.values() if isinstance(v, float)), f"Non-integer float in {d}"
        # None marks an absent kwarg; dropping it gives dense and sparse dataset shapes the same context.
        new_kwargs.append({k: int(v) if isinstance(v, float) else v for k, v in d.items() if v is not None})
    return IFEvalMetricContext(
        key=item["key"],
        instruction_id_list=item["instruction_id_list"],
        prompt=item["prompt"],
        additional_kwargs=new_kwargs,
    )


def _ifeval(
    id: str,
    *,
    dataset_path: str,
    metrics: list[type["BaseMetric"]],
    language: LanguageSpec,
    dataset: DatasetPolicy | None,
) -> Benchmark:
    return ComposedBenchmark.compose(
        id=id,
        kind=Generative(
            build_prompt=lambda item: item["prompt"],
            cue="",  # the model answers directly; no assistant cue
            ground_truth=lambda item: None,  # no gold answer — scored from the context's instruction checks
            metrics=metrics,
            context=_ifeval_context,
        ),
        answer=ExtractFromCompletion(lambda completion_text: completion_text),  # checks run on the whole generation
        sample_split="train",
        fewshot=NoFewShot(),
        dataset_policy=dataset if dataset is not None else pinned_by_framework(dataset_path),
        language=language,
    )


def ifeval(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _ifeval(
        "IFEval", dataset_path=IFEVAL_DATASET_PATH, metrics=[IFEvalMetric], language=Language.ENG, dataset=dataset
    )


def ifeval_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _ifeval(
        "IFEvalDe",
        dataset_path=IFEVAL_DE_DATASET_PATH,
        metrics=[IFEvalMetric, LanguageRawConsistencyChecker],
        language=Language.DEU,
        dataset=dataset,
    )


IFEVAL_BENCHMARKS: list[Benchmark] = [ifeval(), ifeval_de()]
