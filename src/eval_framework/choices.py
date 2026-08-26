"""Choice readers: extracting the fields a choice-based styler needs out of a raw dataset item,
so neither the eval nor the styler has to know a benchmark's item schema."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, final, override

from eval_framework.tasks.task_style import shuffle_correct_with_distractors


@dataclass(frozen=True)
class ChoiceFields:
    """The fields a choice-based styler needs out of a single dataset item.

    ``choices`` and ``correct_index`` are produced together (a benchmark may shuffle the correct
    answer in among distractors), so a reader yields them in one ``read`` rather than via separate
    calls that would each have to re-derive the same shuffle.
    """

    raw_question: str
    choices: list[str]
    correct_index: int


class ChoiceReader(ABC):
    """Reads the fields a styler needs out of a raw dataset item.

    Isolates dataset-schema knowledge here, so neither the eval nor the styler has to know the shape
    of a benchmark's items.
    """

    @abstractmethod
    def read(self, item: dict[str, Any]) -> ChoiceFields: ...


@final
class EllaMindReader(ChoiceReader):
    """Reads an EllaMind-style choice item, shuffling the correct answer in among its distractors.

    The EllaMind datasets share one shape — a question, a correct answer, and a per-level distractor
    set — but differ in field names, and in whether the distractor field holds a single value
    (``singular=True``, wrapped into a one-element list) or already a list.
    """

    def __init__(
        self,
        distractor_level: Literal["easy", "hard"],
        *,
        question_field: str,
        correct_field: str,
        easy_field: str,
        hard_field: str,
        singular: bool = False,
    ) -> None:
        self._distractor_level = distractor_level
        self._question_field = question_field
        self._correct_field = correct_field
        self._easy_field = easy_field
        self._hard_field = hard_field
        self._singular = singular

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        raw = item[self._easy_field if self._distractor_level == "easy" else self._hard_field]
        distractors = [raw] if self._singular else raw
        choices, correct_index = shuffle_correct_with_distractors(
            correct=item[self._correct_field],
            distractors=distractors,
            seed_text=item[self._question_field] + item[self._correct_field],
        )
        return ChoiceFields(raw_question=item[self._question_field], choices=choices, correct_index=correct_index)
