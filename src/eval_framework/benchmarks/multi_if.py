"""Multi-IF: multi-turn, multilingual instruction following (https://arxiv.org/abs/2410.15553).

Each conversation has up to three user turns, each adding IFEval-style constraints on top of the earlier ones. The
model answers every turn with its own earlier answers in context, and each turn is graded against all the
constraints in force at that point. Subjects are the dataset's eight languages; 0-shot only.
"""

import json
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, final, override

from eval_framework.answer import ExtractFromCompletion
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark, Eval, ResponseType, Sample
from eval_framework.eval_kind import Generative
from eval_framework.fewshot import NoFewShot
from eval_framework.metrics.completion.multi_if import MultiIFMetric, MultiIFMetricContext
from eval_framework.shared.types import Completion
from eval_framework.subjects import ListOfSubjects
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from template_formatting.formatter import BaseFormatter, Message, Role

if TYPE_CHECKING:
    from eval_framework.llm.base import BaseLLM
    from eval_framework.metrics.base import BaseMetric

MULTI_IF_DATASET_PATH = "facebook/Multi-IF"

MULTI_IF_LANGUAGES = {
    # Chinese and Hindi exist only among the ISO 639 members added at import, which mypy cannot see.
    "Chinese": Language["ZHO"],
    "English": Language.ENG,
    "French": Language.FRA,
    "Hindi": Language["HIN"],
    "Italian": Language.ITA,
    "Portuguese": Language.POR,
    "Russian": Language.RUS,
    "Spanish": Language.SPA,
}

_MAX_TURNS = 3


def _multi_if_context(item: dict[str, Any]) -> MultiIFMetricContext:
    # Every turn column is a JSON string; a conversation with fewer turns leaves the later ones empty.
    turns = [t for t in range(1, _MAX_TURNS + 1) if item[f"turn_{t}_prompt"]]
    return MultiIFMetricContext(
        key=item["key"],
        language=item["language"],
        turn_prompts=[json.loads(item[f"turn_{t}_prompt"])["content"] for t in turns],
        turn_instruction_id_lists=[json.loads(item[f"turn_{t}_instruction_id_list"]) for t in turns],
        turn_kwargs=[[json.loads(kwargs) for kwargs in json.loads(item[f"turn_{t}_kwargs"])] for t in turns],
    )


@final
class _MultiTurnEval(Eval):
    """Answers each sample's first turn, then feeds every answer back and asks the next turn, until the
    conversation in the sample's context runs out. The returned completion carries the whole conversation."""

    def __init__(self, inner: Eval) -> None:
        self._inner = inner

    @override
    def iterate_samples(self, num_samples: int | None = None) -> Iterable[Sample]:
        return self._inner.iterate_samples(num_samples)

    @override
    def generate_completions(
        self,
        llm: "BaseLLM",
        samples: list[Sample],
        stop_sequences: list[str] | None = None,
        max_tokens: int | None = None,
        fail_on_error: bool = True,
    ) -> list[Completion]:
        # Keyed by batch position: with repeats, one sample id can appear several times in a batch.
        finished: dict[int, Completion] = {}
        pending = list(enumerate(samples))
        turn = 0
        while pending:
            completions = self._inner.generate_completions(
                llm, [sample for _, sample in pending], stop_sequences, max_tokens, fail_on_error
            )
            turn += 1
            still_pending = []
            for (position, sample), completion in zip(pending, completions, strict=True):
                assert isinstance(sample.context, MultiIFMetricContext)
                # The post-processed answer, not the raw one, is what the model sees in later turns.
                history = sample.messages + [Message(role=Role.ASSISTANT, content=completion.completion)]
                if completion.error is not None or turn == len(sample.context.turn_prompts):
                    finished[position] = completion.model_copy(update={"messages": history})
                else:
                    next_turn = Message(role=Role.USER, content=sample.context.turn_prompts[turn])
                    still_pending.append((position, sample.model_copy(update={"messages": history + [next_turn]})))
            pending = still_pending
        return [finished[position] for position in range(len(samples))]

    @override
    def get_metadata(self) -> dict[str, str | list[str]]:
        return self._inner.get_metadata()

    @override
    def get_stop_sequences(self) -> list[str]:
        return self._inner.get_stop_sequences()

    @override
    def get_max_tokens(self) -> int | None:
        return self._inner.get_max_tokens()

    @override
    def get_response_type(self) -> ResponseType:
        return self._inner.get_response_type()

    @override
    def display_name(self) -> str:
        return self._inner.display_name()


@final
class _MultiTurnBenchmark(Benchmark):
    def __init__(self, inner: Benchmark) -> None:
        self._inner = inner

    @override
    def id(self) -> str:
        return self._inner.id()

    @override
    def response_type(self) -> ResponseType:
        return self._inner.response_type()

    @override
    def metrics(self) -> list[type["BaseMetric"]]:
        return self._inner.metrics()

    @override
    def subjects(self, tokens: list[str] | None = None) -> list[str]:
        return self._inner.subjects(tokens)

    @override
    def display_name(self) -> str:
        return self._inner.display_name()

    @override
    def create(
        self,
        num_fewshot: int,
        custom_subjects: list[str] | None,
        custom_hf_revision: str | None,
        seed: int | None = None,
    ) -> Eval:
        return _MultiTurnEval(self._inner.create(num_fewshot, custom_subjects, custom_hf_revision, seed))

    @override
    def markdown_doc(self, formatters: Sequence[BaseFormatter]) -> str:
        return self._inner.markdown_doc(formatters)


def multi_if(dataset: DatasetPolicy | None = None) -> Benchmark:
    kind = Generative(
        build_prompt=lambda item: json.loads(item["turn_1_prompt"])["content"],
        cue="",
        ground_truth=lambda item: None,  # no gold answer: scored from the context's per-turn instruction checks
        metrics=[MultiIFMetric],
        context=_multi_if_context,
    )
    dataset_policy = (
        dataset
        if dataset is not None
        else pinned_by_framework(MULTI_IF_DATASET_PATH).subject_encoded_in_column(config=None, column="language")
    )
    return _MultiTurnBenchmark(
        ComposedBenchmark.compose(
            id="MultiIF",
            display_name="Multi-IF",
            kind=kind,
            answer=ExtractFromCompletion(lambda completion_text: completion_text),
            sample_split="train",
            fewshot=NoFewShot(),
            subjects=ListOfSubjects(list(MULTI_IF_LANGUAGES)),
            dataset_policy=dataset_policy,
            language=MULTI_IF_LANGUAGES,
        )
    )


MULTI_IF_BENCHMARKS: list[Benchmark] = [multi_if()]
