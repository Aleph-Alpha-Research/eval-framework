"""Choice readers: extracting the fields a choice-based styler needs out of a raw dataset item,
so neither the eval nor the styler has to know a benchmark's item schema."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


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
