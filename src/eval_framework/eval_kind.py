from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, final, override

from eval_framework.choices import ChoiceReader
from eval_framework.contract import ResponseType

if TYPE_CHECKING:
    from eval_framework.metrics.base import BaseMetric
    from eval_framework.tasks.task_style import TaskStyler


@dataclass(frozen=True)
class SampleBody:
    prompt: str  # the user turn
    cue: str  # the assistant turn priming the answer; "" for no assistant turn
    possible_completions: list[str]
    ground_truth: str


@dataclass(frozen=True)
class FewshotExample:
    prompt: str  # the user turn
    answer: str  # the assistant turn (the shown correct answer)


class EvalKind(ABC):
    """How a kind of task becomes scored model interactions. Describes what kind of test this is.

    E.g. Multiple choice vs Free Form answers.

    A kind deals only in text; ``ComposedEval`` owns the (fixed) mapping to USER / ASSISTANT turns.
    """

    response_type: ResponseType
    metrics: list[type["BaseMetric"]]

    @abstractmethod
    def fewshot(self, item: dict[str, Any]) -> FewshotExample:
        """One solved few-shot example: the prompt shown and the answer shown."""

    @abstractmethod
    def samples(self, item: dict[str, Any]) -> list[SampleBody]:
        """The scored sample(s) for one eval item — one for most kinds, more when a kind fans out."""

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
        self.response_type = styler.response_type
        self.metrics = styler.metrics

    @override
    def fewshot(self, item: dict[str, Any]) -> FewshotExample:
        fields = self._reader.read(item)
        return FewshotExample(
            prompt=self._styler.get_instruction_text(fields.raw_question, fields.choices),
            answer=self._styler.get_fewshot_target_text(fields.choices, fields.correct_index),
        )

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
