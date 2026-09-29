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
    # A leading SYSTEM turn for this sample, or None for none.
    system_prompt: str | None = None


def assemble_messages(
    fewshot: list[FewshotExample],
    body: SampleBody,
    *,
    initial_prompt: str | None = None,
) -> list[Message]:
    """The standard prompt: an optional SYSTEM turn (from ``body.system_prompt``), the few-shot demonstrations
    as USER / ASSISTANT pairs, then the item's USER turn and (optional) ASSISTANT cue — with ``initial_prompt``
    folded once into the first turn (above the demonstrations). The single assembler every kind's ``messages``
    delegates to."""
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
    if body.system_prompt is not None:
        messages.insert(0, Message(role=Role.SYSTEM, content=body.system_prompt))
    return messages


class EvalKind(ABC):
    """The bound, per-run prompt assembler: the messages to put in front of the model and what its
    candidates/ground truth are.

    A kind owns the full prompt for an item — the few-shot demonstrations, its own USER turn and (optional)
    ASSISTANT cue, and any preamble — assembled by ``messages``; ``ComposedEval`` only supplies the drawn
    few-shot examples and wraps the result.
    """

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


class BenchmarkKind(ABC):
    """The prompt side a benchmark author declares: the scoring metrics, and how the user-supplied run
    arguments bind into the per-run assembler.

    E.g. multiple choice vs free-form answers. The answer side (response type, generation bounds,
    extraction) is an injected ``AnswerPolicy``.
    """

    @abstractmethod
    def metrics(self) -> list[type["BaseMetric"]]:
        """The metrics this kind is scored with."""

    @abstractmethod
    def bind(self, *, user_prompt_suffix: str | None) -> EvalKind:
        """The benchmark -> eval transition: fold the user-supplied arguments this kind is concerned with
        into the per-run assembler, or reject them. ``user_prompt_suffix`` steers a reasoning model's
        thinking budget through a control token on the evaluated user turn (e.g. ``/think_short``); a kind
        scored by loglikelihood over fixed candidates generates nothing — there is no trace to steer — and
        raises."""


def reject_user_prompt_suffix(user_prompt_suffix: str | None) -> None:
    """The ``bind`` guard of a kind the model never completes free-form: any requested suffix is an error."""
    if user_prompt_suffix is not None:
        raise ValueError("user_prompt_suffix is only supported for completion tasks.")


@final
class WithUserPromptSuffix(EvalKind):
    """An assembler with the user-supplied suffix appended verbatim to the evaluated user turn — the last
    USER message of the inner assembly (the few-shot turns precede it; the cue is an ASSISTANT turn).

    Appending after assembly matches ``BaseTask``: an initial prompt folds into the *first* message, so the
    suffix ends the evaluated turn either way."""

    def __init__(self, inner: EvalKind, suffix: str) -> None:
        self._inner = inner
        self._suffix = suffix

    @override
    def samples(self, item: dict[str, Any]) -> list[SampleBody]:
        return self._inner.samples(item)

    @override
    def messages(self, body: SampleBody, *, fewshot: list[FewshotExample], subject_label: str) -> list[Message]:
        messages = self._inner.messages(body, fewshot=fewshot, subject_label=subject_label)
        for index in reversed(range(len(messages))):
            if messages[index].role == Role.USER:
                suffixed = Message(role=Role.USER, content=f"{messages[index].content}{self._suffix}")
                return [*messages[:index], suffixed, *messages[index + 1 :]]
        raise ValueError("Cannot append user_prompt_suffix: the assembled prompt contains no user message.")

    @override
    def metadata(self) -> dict[str, str]:
        return self._inner.metadata()


@final
class Choice(BenchmarkKind, EvalKind):
    """Choice-based eval kind: wraps a reader (item -> ChoiceFields) and a styler (multiple-choice /
    cloze / BPB), producing exactly one scored sample per item."""

    def __init__(self, reader: ChoiceReader, styler: "TaskStyler") -> None:
        self._reader = reader
        self._styler = styler

    @override
    def bind(self, *, user_prompt_suffix: str | None) -> EvalKind:
        # The candidates are scored by loglikelihood; the model never completes the prompt free-form.
        reject_user_prompt_suffix(user_prompt_suffix)
        return self

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
class Generative(BenchmarkKind, EvalKind):
    """Free-form question -> answer kind: one sample per item, no scored candidates (the answer is extracted
    from the generation by the injected ``AnswerPolicy``). ``build_prompt`` frames the question, ``cue``
    primes the answer turn (``""`` for none), ``ground_truth`` derives the gold answer, and ``metrics`` are
    the scoring metrics. ``context`` derives the per-sample scoring material a metric needs beyond the gold
    string (see ``SampleBody.context``); ``initial_prompt`` is a preamble prepended once above the first
    (few-shot) turn, and ``system_prompt`` derives a leading SYSTEM turn per item (``None`` for no system
    turn — e.g. an instruction-following task that carries its constraints in the system prompt)."""

    def __init__(
        self,
        *,
        build_prompt: ItemText,
        cue: str,
        ground_truth: ItemGroundTruth,
        metrics: list[type["BaseMetric"]],
        context: ItemContext | None = None,
        initial_prompt: str | None = None,
        system_prompt: ItemText | None = None,
    ) -> None:
        self._build_prompt = build_prompt
        self._cue = cue
        self._ground_truth = ground_truth
        self._metrics = metrics
        self._context: ItemContext = context if context is not None else NoContext
        self._initial_prompt = initial_prompt
        self._system_prompt = system_prompt

    @override
    def bind(self, *, user_prompt_suffix: str | None) -> EvalKind:
        # Free-form completion supports the suffix; it lands on the evaluated user turn.
        if user_prompt_suffix is None:
            return self
        return WithUserPromptSuffix(self, user_prompt_suffix)

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
                system_prompt=self._system_prompt(item) if self._system_prompt is not None else None,
            )
        ]

    @override
    def messages(self, body: SampleBody, *, fewshot: list[FewshotExample], subject_label: str) -> list[Message]:
        return assemble_messages(fewshot, body, initial_prompt=self._initial_prompt)
