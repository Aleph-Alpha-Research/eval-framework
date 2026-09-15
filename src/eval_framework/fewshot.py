"""Few-shot policies: where a composed eval draws its demonstrations from, how they are rendered — and
whether it draws any.

A ``FewShot`` owns the *source* of demonstrations (which split, sampled leak-safely) and their *rendering*
into solved prompt/answer pairs; the eval only wraps those into USER / ASSISTANT turns. ``NoFewShot`` lets
a benchmark declare "0-shot only" structurally, so the constraint is enforced at creation instead of via a
placeholder split.
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, final, override

from eval_framework.choices import ChoiceReader

if TYPE_CHECKING:
    from eval_framework.tasks.task_style import TaskStyler


@dataclass(frozen=True)
class FewshotExample:
    prompt: str  # the user turn
    answer: str  # the assistant turn (the shown correct answer)


class FewShot(ABC):
    """The source of a composed eval's few-shot demonstrations, their rendering, and whether it permits any."""

    @abstractmethod
    def split(self) -> str | None:
        """The dataset split demonstrations are drawn from (retained when the dataset loads), or None
        when the policy draws none."""

    @abstractmethod
    def check(self, num_fewshot: int) -> None:
        """Raise if ``num_fewshot`` is incompatible with this policy. Called when the eval is created,
        so an unsupported request fails before any dataset is touched."""

    @abstractmethod
    def examples(
        self,
        dataset: dict[str, list[dict[str, Any]]],
        *,
        sample_split: str,
        item: dict[str, Any],
        num_fewshot: int,
        rnd: random.Random,
    ) -> list["FewshotExample"]:
        """The rendered demonstrations to show before ``item`` — sampled leak-safely (never ``item``
        itself) and formatted into solved prompt/answer pairs."""

    @abstractmethod
    def metadata(self) -> dict[str, str]:
        """Few-shot metadata merged into the eval's ``get_metadata`` (e.g. the source split)."""


@final
class SampledFewShot(FewShot):
    """Draws ``num_fewshot`` demonstrations at random from ``split`` and renders each with ``reader`` +
    ``styler`` — the shown prompt, then the correct answer. When ``split`` is also the sample split, the
    current eval item is excluded so its own answer never leaks into its prompt."""

    def __init__(self, reader: ChoiceReader, styler: "TaskStyler", split: str) -> None:
        self._reader = reader
        self._styler = styler
        self._split = split

    @override
    def split(self) -> str | None:
        return self._split

    @override
    def check(self, num_fewshot: int) -> None:
        return  # any shot count is supported

    @override
    def examples(
        self,
        dataset: dict[str, list[dict[str, Any]]],
        *,
        sample_split: str,
        item: dict[str, Any],
        num_fewshot: int,
        rnd: random.Random,
    ) -> list[FewshotExample]:
        sampled = self._sample(dataset, sample_split=sample_split, item=item, num_fewshot=num_fewshot, rnd=rnd)
        return [self._render(demonstration) for demonstration in sampled]

    def _sample(
        self,
        dataset: dict[str, list[dict[str, Any]]],
        *,
        sample_split: str,
        item: dict[str, Any],
        num_fewshot: int,
        rnd: random.Random,
    ) -> list[dict[str, Any]]:
        if num_fewshot <= 0:
            return []
        fewshot_pool = dataset[self._split]
        if self._split == sample_split:
            # Same split for demonstrations and evaluation: over-sample by one, drop the current item
            # if it was drawn (so its answer never leaks), then truncate back to num_fewshot.
            drawn = rnd.sample(fewshot_pool, num_fewshot + 1)
            drawn = [demonstration for demonstration in drawn if demonstration != item]
            return drawn[:num_fewshot]
        # Separate splits: no risk of leaking the current item, sample directly.
        return rnd.sample(fewshot_pool, num_fewshot)

    def _render(self, item: dict[str, Any]) -> FewshotExample:
        fields = self._reader.read(item)
        return FewshotExample(
            prompt=self._styler.get_instruction_text(fields.raw_question, fields.choices),
            answer=self._styler.get_fewshot_target_text(fields.choices, fields.correct_index),
        )

    @override
    def metadata(self) -> dict[str, str]:
        return {"fewshot_split": self._split}


@final
class NoFewShot(FewShot):
    """A benchmark that only runs 0-shot: it names no source split and rejects any few-shot request."""

    @override
    def split(self) -> str | None:
        return None

    @override
    def check(self, num_fewshot: int) -> None:
        if num_fewshot != 0:
            raise ValueError(f"This benchmark is 0-shot only; num_fewshot must be 0, got {num_fewshot}.")

    @override
    def examples(
        self,
        dataset: dict[str, list[dict[str, Any]]],
        *,
        sample_split: str,
        item: dict[str, Any],
        num_fewshot: int,
        rnd: random.Random,
    ) -> list[FewshotExample]:
        return []

    @override
    def metadata(self) -> dict[str, str]:
        return {}
