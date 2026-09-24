from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, final, override

from eval_framework.choices import ChoiceReader
from eval_framework.fewshot import FewshotExample
from eval_framework.shared.types import BaseMetricContext
from template_formatting.formatter import Message, Role

if TYPE_CHECKING:
    from eval_framework.metrics.base import BaseMetric
    from eval_framework.tasks.task_style import TaskStyler


@dataclass(frozen=True)
class SampleBody:
    prompt: str  # the user turn
    cue: str  # the assistant turn priming the answer; "" for no assistant turn
    possible_completions: list[str]
    # A single gold answer, several equally-correct ones (open-QA), or None when the task has no gold and is
    # scored purely from the context (e.g. IFEval's instruction checks).
    ground_truth: str | list[str] | None
    # Per-sample material the metric (or answer extraction) needs beyond the prompt/completion/ground_truth:
    # gold answer structure for F1, an instruction-following spec, a code test harness. None for most kinds.
    context: BaseMetricContext | list[BaseMetricContext] | None = None


def assemble_messages(
    fewshot: list[FewshotExample],
    body: SampleBody,
    *,
    system_prompt: str | None = None,
    initial_prompt: str | None = None,
) -> list[Message]:
    """The standard prompt: an optional SYSTEM turn, the few-shot demonstrations as USER / ASSISTANT pairs,
    then the item's USER turn and (optional) ASSISTANT cue — with ``initial_prompt`` folded once into the
    first turn (above the demonstrations). The single assembler every kind's ``messages`` delegates to."""
    messages: list[Message] = []
    for example in fewshot:
        messages.append(Message(role=Role.USER, content=example.prompt))
        messages.append(Message(role=Role.ASSISTANT, content=example.answer))
    messages.append(Message(role=Role.USER, content=body.prompt))
    if initial_prompt is not None:
        first = messages[0]
        messages[0] = Message(role=first.role, content=f"{initial_prompt}\n\n{first.content}")
    if body.cue:
        messages.append(Message(role=Role.ASSISTANT, content=body.cue))
    if system_prompt is not None:
        messages.insert(0, Message(role=Role.SYSTEM, content=system_prompt))
    return messages


class EvalKind(ABC):
    """The prompt side of a task: the messages to put in front of the model and what its candidates/ground
    truth are.

    E.g. Multiple choice vs Free Form answers. A kind owns the full prompt for an item — the few-shot
    demonstrations, its own USER turn and (optional) ASSISTANT cue, and any preamble — assembled by
    ``messages``; ``ComposedEval`` only supplies the drawn few-shot examples and wraps the result. The answer
    side (response type, generation bounds, extraction) is an injected ``AnswerPolicy``.
    """

    @abstractmethod
    def metrics(self) -> list[type["BaseMetric"]]:
        """The metrics this kind is scored with."""

    @abstractmethod
    def samples(self, item: dict[str, Any]) -> list[SampleBody]:
        """The scored sample(s) for one eval item — one for most kinds, more when a kind fans out."""

    @abstractmethod
    def messages(self, body: SampleBody, *, fewshot: list[FewshotExample], subject_label: str) -> list[Message]:
        """The full message list for one sample — typically ``assemble_messages`` with the kind's own preamble
        and/or system prompt."""

    def metadata(self) -> dict[str, str]:
        """Kind-specific metadata merged into the eval's ``get_metadata`` (e.g. the task style)."""
        return {}


@final
class Choice(EvalKind):
    """Choice-based eval kind: wraps a reader (item -> ChoiceFields) and a styler (multiple-choice /
    cloze / BPB), producing exactly one scored sample per item."""

    def __init__(self, reader: ChoiceReader, styler: "TaskStyler") -> None:
        self._reader = reader
        self._styler = styler

    @override
    def metrics(self) -> list[type["BaseMetric"]]:
        return self._styler.metrics

    @override
    def samples(self, item: dict[str, Any]) -> list[SampleBody]:
        fields = self._reader.read(item)
        completions = self._styler.get_possible_completions(fields.choices, fields.correct_index)
        assert completions is not None  # choice stylers always score a candidate list
        return [
            SampleBody(
                prompt=self._styler.get_instruction_text(fields.raw_question, fields.choices),
                cue=self._styler.get_cue_text(),
                possible_completions=completions,
                ground_truth=self._styler.get_ground_truth(fields.choices, fields.correct_index),
            )
        ]

    @override
    def metadata(self) -> dict[str, str]:
        return self._styler.get_extra_metadata()

    @override
    def messages(self, body: SampleBody, *, fewshot: list[FewshotExample], subject_label: str) -> list[Message]:
        return assemble_messages(fewshot, body, initial_prompt=self._styler.initial_prompt(subject_label))


# item -> a rendered prompt / cue string.
ItemText = Callable[[dict[str, Any]], str]

# item -> the gold answer: one string, several equally-correct ones, or None (no gold; scored via context).
ItemGroundTruth = Callable[[dict[str, Any]], str | list[str] | None]

# item -> the per-sample metric context (gold structure / test harness / instruction spec), or None.
ItemContext = Callable[[dict[str, Any]], BaseMetricContext | list[BaseMetricContext] | None]


def NoContext(item: dict[str, Any]) -> None:
    """The default ``ItemContext``: the sample carries no metric context."""
    return None


@final
class Generative(EvalKind):
    """Free-form question -> answer kind: one sample per item, no scored candidates (the answer is extracted
    from the generation by the injected ``AnswerPolicy``). ``build_prompt`` frames the question, ``cue``
    primes the answer turn (``""`` for none), ``ground_truth`` derives the gold answer, and ``metrics`` are
    the scoring metrics. ``context`` derives the per-sample scoring material a metric needs beyond the gold
    string (see ``SampleBody.context``); ``initial_prompt`` is a preamble prepended once above the first
    (few-shot) turn, and ``system_prompt`` is a leading SYSTEM turn."""

    def __init__(
        self,
        *,
        build_prompt: ItemText,
        cue: str,
        ground_truth: ItemGroundTruth,
        metrics: list[type["BaseMetric"]],
        context: ItemContext | None = None,
        initial_prompt: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._build_prompt = build_prompt
        self._cue = cue
        self._ground_truth = ground_truth
        self._metrics = metrics
        self._context: ItemContext = context if context is not None else NoContext
        self._initial_prompt = initial_prompt
        self._system_prompt = system_prompt

    @override
    def metrics(self) -> list[type["BaseMetric"]]:
        return self._metrics

    @override
    def samples(self, item: dict[str, Any]) -> list[SampleBody]:
        return [
            SampleBody(
                prompt=self._build_prompt(item),
                cue=self._cue,
                possible_completions=[],
                ground_truth=self._ground_truth(item),
                context=self._context(item),
            )
        ]

    @override
    def messages(self, body: SampleBody, *, fewshot: list[FewshotExample], subject_label: str) -> list[Message]:
        return assemble_messages(fewshot, body, system_prompt=self._system_prompt, initial_prompt=self._initial_prompt)
