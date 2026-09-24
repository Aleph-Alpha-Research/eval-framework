"""Natural Questions (open): https://huggingface.co/datasets/google-research-datasets/nq_open

Short factoid questions, each with one or more equally-correct gold answers. ``NaturalQsOpen`` generates a
free-form answer scored by DROP F1 / exact match (against every gold answer, carried as a per-sample
``DropMetricContext``); ``NaturalQsOpenMC_OLMES`` scores labelled candidate answers by loglikelihood
(allenai/nq-gen2mc).
"""

from typing import Any, final, override

from eval_framework.answer import ExtractFromCompletion
from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.eval_kind import Generative
from eval_framework.fewshot import FewShot, FewshotExample, FewShotSplit, FunctionRenderer
from eval_framework.metrics.completion.drop_completion import DropF1ExactMatch, DropMetricContext
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import MCStyle, answer_key_to_index

NQ_OPEN_DATASET_PATH = "google-research-datasets/nq_open"
NQ_GEN2MC_DATASET_PATH = "allenai/nq-gen2mc"

# Generation stops at the next question or a blank line (the boundary between few-shot blocks).
_OPEN_STOP_SEQUENCES = ["Question:", "Q:", "\n\n"]


def _open_prompt(item: dict[str, Any]) -> str:
    return f"Question: {item.get('question', '')}\n"


def _open_ground_truth(item: dict[str, Any]) -> list[str]:
    return [f" {a}" for a in item.get("answer", [])]


def _open_context(item: dict[str, Any]) -> DropMetricContext | None:
    # DROP F1 scores the generation against every gold answer (each a single-span tuple).
    answers = item.get("answer", [])
    if not answers:
        return None
    return DropMetricContext(answer_tuples=[[a] for a in answers])


def _open_demo(item: dict[str, Any]) -> FewshotExample:
    # OLMES joins the (already space-prefixed) gold answers with a comma for the demonstration target.
    return FewshotExample(prompt=_open_prompt(item), answer=f"Answer:{','.join(_open_ground_truth(item))}")


def natural_qs_open(dataset: DatasetPolicy | None = None) -> Benchmark:
    return ComposedBenchmark.compose(
        id="NaturalQsOpen",
        kind=Generative(
            build_prompt=_open_prompt,
            cue="Answer:",  # the model continues after the cue
            ground_truth=_open_ground_truth,
            metrics=[DropF1ExactMatch],
            context=_open_context,
        ),
        answer=ExtractFromCompletion(
            lambda completion_text: completion_text,  # F1 scores the whole generation; nothing is extracted
            _OPEN_STOP_SEQUENCES,
            max_tokens=50,
        ),
        sample_split="validation",
        fewshot=FewShot(FewShotSplit("train"), FunctionRenderer(_open_demo)),
        dataset_policy=dataset if dataset is not None else pinned_by_framework(NQ_OPEN_DATASET_PATH),
        language=Language.ENG,
    )


@final
class _NaturalQsChoiceReader(ChoiceReader):
    """Reads the nq-gen2mc question/choices; the correct index is decoded from ``answerKey``."""

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        return ChoiceFields(
            raw_question=item.get("question", ""),
            choices=item.get("choices", {}).get("text", []),
            correct_index=answer_key_to_index(item.get("answerKey", "")),
        )


def natural_qs_open_mc_olmes(dataset: DatasetPolicy | None = None) -> Benchmark:
    """OLMES lays out the options with a leading space (" A. ...")."""
    return ComposedBenchmark.choice(
        id="NaturalQsOpenMC_OLMES",
        reader=_NaturalQsChoiceReader(),
        styler=MCStyle(space_prefixed_labels=True),
        sample_split="validation",
        fewshot_split="validation",
        dataset_policy=dataset if dataset is not None else pinned_by_framework(NQ_GEN2MC_DATASET_PATH),
        language=Language.ENG,
    )


NATURALQS_OPEN_BENCHMARKS: list[Benchmark] = [natural_qs_open(), natural_qs_open_mc_olmes()]
