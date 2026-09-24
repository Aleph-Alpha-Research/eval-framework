"""Few-shot policies: what demonstrations a composed eval shows before each item, and how they render."""

import logging
import random
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, final, override

from eval_framework.choices import ChoiceReader

if TYPE_CHECKING:
    from eval_framework.tasks.task_style import TaskStyler

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FewshotExample:
    prompt: str  # the user turn
    answer: str  # the assistant turn (the shown correct answer)


# ---------------------------------------------------------------------------
# Rendering: one drawn row -> a demonstration
# ---------------------------------------------------------------------------


class FewShotRenderer(ABC):
    """Turns one drawn dataset row into a solved demonstration — the shown prompt and its correct answer.

    The *rendering* half of few-shot, orthogonal to *which rows* a source provides."""

    @abstractmethod
    def render(self, item: dict[str, Any]) -> FewshotExample:
        """Render one dataset row into a demonstration (prompt + shown answer)."""


@final
class ChoiceRenderer(FewShotRenderer):
    """Renders through the same choice ``reader`` + ``styler`` that score the task, so the shots look exactly
    like the scored prompt (used by every choice / loglikelihood benchmark)."""

    def __init__(self, reader: ChoiceReader, styler: "TaskStyler") -> None:
        self._reader = reader
        self._styler = styler

    @override
    def render(self, item: dict[str, Any]) -> FewshotExample:
        fields = self._reader.read(item)
        return FewshotExample(
            prompt=self._styler.get_instruction_text(fields.raw_question, fields.choices),
            answer=self._styler.get_fewshot_target_text(fields.choices, fields.correct_index),
        )


@final
class FunctionRenderer(FewShotRenderer):
    """Renders via a benchmark-supplied ``item -> FewshotExample`` function — for generative tasks that build
    the demonstration directly rather than through a choice styler."""

    def __init__(self, render: Callable[[dict[str, Any]], FewshotExample]) -> None:
        self._render = render

    @override
    def render(self, item: dict[str, Any]) -> FewshotExample:
        return self._render(item)


# ---------------------------------------------------------------------------
# Sourcing: where the demonstration rows come from
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FewShotDoc:
    """What ``markdown_doc`` needs to describe a source without running it: the demonstration split (``None``
    for a fixed block or no few-shot) and how many demonstrations to show in the rendered example."""

    split: str | None
    example_shots: int


class FewShotSource(ABC):
    """Where a few-shot policy draws its demonstration rows, and how many. An immutable spec — the per-run
    pool and count live on the generator, so a source instance is safe to share across evals.

    ``build_pool`` (once, when data is loaded) and ``draw`` (per eval item) are split so a candidate pool is
    filtered/materialised once rather than per item."""

    @abstractmethod
    def resolve_count(self, num_fewshot: int) -> int:
        """The effective shot count — usually ``num_fewshot`` unchanged; a fixed source pins it."""

    @abstractmethod
    def build_pool(
        self, dataset: Mapping[str, Any], *, sample_split: str, sample_rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """The candidate demonstration rows, from the already-loaded data (called once per subject)."""

    @abstractmethod
    def draw(
        self, pool: list[dict[str, Any]], *, item: dict[str, Any], count: int, rnd: random.Random
    ) -> list[dict[str, Any]]:
        """Select ``count`` rows from ``pool`` to show before ``item``."""

    @abstractmethod
    def metadata(self, sample_split: str) -> dict[str, str]:
        """Source metadata merged into the eval's ``get_metadata`` (e.g. the demonstration split)."""

    @abstractmethod
    def documentation(self, sample_split: str) -> FewShotDoc:
        """The demonstration split and example shot count for the rendered task docs."""


@final
class SampleSplit(FewShotSource):
    """Draws demonstrations from the eval (sample) split itself — leak-safe: the current item is excluded so
    its own answer never appears in its prompt. Optionally restricted to rows passing ``keep``."""

    def __init__(self, *, keep: Callable[[dict[str, Any]], bool] | None = None) -> None:
        self._keep = keep

    @override
    def resolve_count(self, num_fewshot: int) -> int:
        return num_fewshot

    @override
    def build_pool(
        self, dataset: Mapping[str, Any], *, sample_split: str, sample_rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return [row for row in sample_rows if self._keep(row)] if self._keep is not None else sample_rows

    @override
    def draw(
        self, pool: list[dict[str, Any]], *, item: dict[str, Any], count: int, rnd: random.Random
    ) -> list[dict[str, Any]]:
        # Over-sample by one and drop the current item if it was drawn, so its answer never leaks.
        oversampled = rnd.sample(pool, count + 1)
        return [row for row in oversampled if row != item][:count]

    @override
    def metadata(self, sample_split: str) -> dict[str, str]:
        return {"fewshot_split": sample_split}

    @override
    def documentation(self, sample_split: str) -> FewShotDoc:
        return FewShotDoc(split=sample_split, example_shots=1)


@final
class FewShotSplit(FewShotSource):
    """Draws demonstrations from a dedicated ``split`` separate from the eval split — no leak check, since the
    pool never overlaps the eval items. Optionally restricted to rows passing ``keep``."""

    def __init__(self, split: str, *, keep: Callable[[dict[str, Any]], bool] | None = None) -> None:
        self._split = split
        self._keep = keep

    @override
    def resolve_count(self, num_fewshot: int) -> int:
        return num_fewshot

    @override
    def build_pool(
        self, dataset: Mapping[str, Any], *, sample_split: str, sample_rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        assert self._split != sample_split, (
            f"FewShotSplit({self._split!r}) names the sample split; use SampleSplit() so draws stay leak-safe."
        )
        rows = list(dataset[self._split])
        return [row for row in rows if self._keep(row)] if self._keep is not None else rows

    @override
    def draw(
        self, pool: list[dict[str, Any]], *, item: dict[str, Any], count: int, rnd: random.Random
    ) -> list[dict[str, Any]]:
        return rnd.sample(pool, count)

    @override
    def metadata(self, sample_split: str) -> dict[str, str]:
        return {"fewshot_split": self._split}

    @override
    def documentation(self, sample_split: str) -> FewShotDoc:
        return FewShotDoc(split=self._split, example_shots=1)


@final
class Predefined(FewShotSource):
    """A fixed, hand-written block of demonstrations (not drawn from the dataset). The count is pinned to
    ``count`` — a benchmark whose prompt uses a canonical fixed few-shot block — warning (rather than showing
    a different number) if a different count is requested."""

    def __init__(self, items: list[dict[str, Any]], *, count: int, label: str) -> None:
        self._items = items
        self._count = count
        self._label = label

    @override
    def resolve_count(self, num_fewshot: int) -> int:
        if num_fewshot != self._count:
            logger.warning(f"{self._label} uses a fixed num_fewshot of {self._count}. Got {num_fewshot}.")
        return self._count

    @override
    def build_pool(
        self, dataset: Mapping[str, Any], *, sample_split: str, sample_rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return self._items

    @override
    def draw(
        self, pool: list[dict[str, Any]], *, item: dict[str, Any], count: int, rnd: random.Random
    ) -> list[dict[str, Any]]:
        return pool[:count]

    @override
    def metadata(self, sample_split: str) -> dict[str, str]:
        return {"fewshot_split": "predefined"}

    @override
    def documentation(self, sample_split: str) -> FewShotDoc:
        return FewShotDoc(split=None, example_shots=self._count)


# ---------------------------------------------------------------------------
# Policy (spec) -> generator (per-run worker)
# ---------------------------------------------------------------------------


class FewShotPolicy(ABC):
    """The immutable few-shot spec a benchmark holds; binds a run's shot count into a generator (mirrors
    ``DatasetPolicy`` → ``DatasetLoader``)."""

    @abstractmethod
    def bind(self, num_fewshot: int) -> "FewShotGenerator":
        """Resolve the shot count — failing fast on an unsupported request, or pinning it for a fixed source —
        and produce the per-run generator that holds it. Called at eval creation, before any load."""

    @abstractmethod
    def documentation(self, sample_split: str) -> FewShotDoc:
        """The demonstration split and example shot count for the rendered task docs."""


class FewShotGenerator(ABC):
    """A per-run few-shot worker, bound to a shot count: it remembers its demonstration pool once the data is
    loaded, then renders the demonstrations to show before each eval item."""

    @abstractmethod
    def prepare(self, dataset: Mapping[str, Any], *, sample_split: str, sample_rows: list[dict[str, Any]]) -> None:
        """Materialise the demonstration pool from the already-loaded ``dataset`` (called once per subject)."""

    @abstractmethod
    def for_item(self, item: dict[str, Any], rnd: random.Random) -> list[FewshotExample]:
        """The rendered demonstrations to show before ``item`` — leak-safe against ``item`` itself."""

    @abstractmethod
    def metadata(self, sample_split: str) -> dict[str, str]:
        """Few-shot metadata merged into the eval's ``get_metadata`` (e.g. the source split)."""


@final
class FewShot(FewShotPolicy):
    """Draw demonstrations from a ``source`` and render each with a ``renderer`` — the two orthogonal axes of
    few-shot (which rows, and how they look)."""

    def __init__(self, source: FewShotSource, renderer: FewShotRenderer) -> None:
        self._source = source
        self._renderer = renderer

    @override
    def bind(self, num_fewshot: int) -> FewShotGenerator:
        return _Generator(self._source.resolve_count(num_fewshot), source=self._source, renderer=self._renderer)

    @override
    def documentation(self, sample_split: str) -> FewShotDoc:
        return self._source.documentation(sample_split)


@final
class _Generator(FewShotGenerator):
    def __init__(self, count: int, *, source: FewShotSource, renderer: FewShotRenderer) -> None:
        self._count = count
        self._source = source
        self._renderer = renderer
        self._pool: list[dict[str, Any]] = []

    @override
    def prepare(self, dataset: Mapping[str, Any], *, sample_split: str, sample_rows: list[dict[str, Any]]) -> None:
        # A 0-shot run draws nothing, so the pool (and any separate few-shot split) need not be built.
        if self._count <= 0:
            return
        self._pool = self._source.build_pool(dataset, sample_split=sample_split, sample_rows=sample_rows)

    @override
    def for_item(self, item: dict[str, Any], rnd: random.Random) -> list[FewshotExample]:
        if self._count <= 0:
            return []
        rows = self._source.draw(self._pool, item=item, count=self._count, rnd=rnd)
        return [self._renderer.render(row) for row in rows]

    @override
    def metadata(self, sample_split: str) -> dict[str, str]:
        return self._source.metadata(sample_split)


@final
class NoFewShot(FewShotPolicy):
    """A benchmark that only runs 0-shot: it rejects any few-shot request and shows no demonstrations."""

    @override
    def bind(self, num_fewshot: int) -> FewShotGenerator:
        if num_fewshot != 0:
            raise ValueError(f"This benchmark is 0-shot only; num_fewshot must be 0, got {num_fewshot}.")
        return _NoGenerator()

    @override
    def documentation(self, sample_split: str) -> FewShotDoc:
        return FewShotDoc(split=None, example_shots=0)


@final
class _NoGenerator(FewShotGenerator):
    @override
    def prepare(self, dataset: Mapping[str, Any], *, sample_split: str, sample_rows: list[dict[str, Any]]) -> None:
        return None

    @override
    def for_item(self, item: dict[str, Any], rnd: random.Random) -> list[FewshotExample]:
        return []

    @override
    def metadata(self, sample_split: str) -> dict[str, str]:
        return {}
