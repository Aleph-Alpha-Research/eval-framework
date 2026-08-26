"""Subjects: the slices a benchmark's evaluation partitions into — which dataset config each loads,
and how each is labelled in samples, metadata, and result aggregation.

A ``SubjectsSelector`` turns the ``--task-subjects`` selector tokens a run requests into the concrete
``Subjects`` to evaluate (an empty token list means "all")."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import final, override


@dataclass(frozen=True)
class Subject:
    """One evaluation slice.

    ``load_key`` selects the dataset config to load (``None`` means the dataset's single config);
    ``label`` identifies the slice in samples, metadata, and result aggregation.
    """

    load_key: str | None
    label: str


Subjects = Sequence[Subject]


class SubjectsSelector(ABC):
    """Selects which slices a run evaluates from its ``--task-subjects`` tokens; ``[]`` selects all."""

    @abstractmethod
    def select(self, tokens: list[str]) -> Subjects: ...


@final
class NoSubject(SubjectsSelector):
    """A task with no subjects: a single unnamed slice. Any selector other than ``"*"`` is an error."""

    @override
    def select(self, tokens: list[str]) -> Subjects:
        if tokens and tokens != ["*"]:
            raise ValueError("This task has no subjects; drop --task-subjects (or pass '*').")
        return (Subject(load_key=None, label="no_subject"),)


@final
class ListOfSubjects(SubjectsSelector):
    """A task whose subjects are named dataset configs. Each name is both the config to load and the
    slice's label; a selector picks names exactly, or ``"*"`` picks all."""

    def __init__(self, names: list[str]) -> None:
        self._names = names

    @override
    def select(self, tokens: list[str]) -> Subjects:
        if tokens:
            unknown = [token for token in tokens if token != "*" and token not in self._names]
            if unknown:
                raise ValueError(f"Unknown subject(s) {unknown}; this task's subjects are {self._names}.")
            wildcard = "*" in tokens
            names = [name for name in self._names if wildcard or name in tokens]
        else:
            names = self._names
        return tuple(Subject(load_key=name, label=name) for name in names)
