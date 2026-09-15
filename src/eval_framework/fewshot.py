"""Few-shot policies: where a composed eval draws its demonstrations from — and whether it draws any.

A ``FewShot`` owns only the *source* of demonstrations (which split, sampled leak-safely) and whether
few-shot is permitted at all. The eval renders each drawn item through its ``EvalKind``; the two concerns
are injected side by side into ``ComposedEval``. ``NoFewShot`` lets a benchmark declare "0-shot only"
structurally, so the constraint is enforced at creation instead of via a placeholder split.
"""

import random
from abc import ABC, abstractmethod
from typing import Any, final, override


class FewShot(ABC):
    """The source of a composed eval's few-shot demonstrations, and whether it permits any."""

    @abstractmethod
    def split(self) -> str | None:
        """The dataset split demonstrations are drawn from (retained when the dataset loads), or None
        when the policy draws none."""

    @abstractmethod
    def check(self, num_fewshot: int) -> None:
        """Raise if ``num_fewshot`` is incompatible with this policy. Called when the eval is created,
        so an unsupported request fails before any dataset is touched."""

    @abstractmethod
    def select(
        self,
        dataset: dict[str, list[dict[str, Any]]],
        *,
        sample_split: str,
        item: dict[str, Any],
        num_fewshot: int,
        rnd: random.Random,
    ) -> list[dict[str, Any]]:
        """The demonstration items to show before ``item`` — already sampled, and never ``item`` itself."""

    @abstractmethod
    def metadata(self) -> dict[str, str]:
        """Few-shot metadata merged into the eval's ``get_metadata`` (e.g. the source split)."""


@final
class SampledFewShot(FewShot):
    """Draws ``num_fewshot`` demonstrations at random from ``split``. When ``split`` is also the sample
    split, the current eval item is excluded so its own answer never leaks into its prompt."""

    def __init__(self, split: str) -> None:
        self._split = split

    @override
    def split(self) -> str | None:
        return self._split

    @override
    def check(self, num_fewshot: int) -> None:
        return  # any shot count is supported

    @override
    def select(
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
            examples = rnd.sample(fewshot_pool, num_fewshot + 1)
            examples = [example for example in examples if example != item]
            return examples[:num_fewshot]
        # Separate splits: no risk of leaking the current item, sample directly.
        return rnd.sample(fewshot_pool, num_fewshot)

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
    def select(
        self,
        dataset: dict[str, list[dict[str, Any]]],
        *,
        sample_split: str,
        item: dict[str, Any],
        num_fewshot: int,
        rnd: random.Random,
    ) -> list[dict[str, Any]]:
        return []

    @override
    def metadata(self) -> dict[str, str]:
        return {}
