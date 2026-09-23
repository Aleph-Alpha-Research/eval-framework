"""Few-shot policies: what demonstrations a composed eval shows before each item, and how they render.

Two phases, mirroring the dataset layer (``DatasetPolicy`` → ``DatasetLoader``):

- ``FewShotPolicy`` is the immutable spec a benchmark holds. ``bind(num_fewshot)`` resolves the run's shot
  count (failing fast, or pinning it for a fixed-shot policy) and produces a ``FewShotGenerator``.
- ``FewShotGenerator`` is the per-run worker: it ``prepare``s its demonstration pool once the data is loaded,
  then renders the demonstrations ``for_item`` at eval time — so ``ComposedEval`` never carries the shot count.

Orthogonal to both is ``FewShotRenderer``: *how* a drawn row becomes a demonstration (a choice reader +
styler, or a plain function). A policy pairs a source (sampled split / fixed block / none) with a renderer;
the eval only wraps the rendered pairs into USER / ASSISTANT turns.
"""

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

    The *rendering* half of few-shot, orthogonal to *which rows* a policy sources."""

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
# Sourcing: policy (spec) -> generator (per-run worker)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FewShotDoc:
    """What ``markdown_doc`` needs to describe a policy without running it: the demonstration split (``None``
    for a fixed block or no few-shot) and how many demonstrations to show in the rendered example."""

    split: str | None
    example_shots: int


class FewShotPolicy(ABC):
    """The immutable few-shot spec a benchmark holds; binds a run's shot count into a generator (mirrors
    ``DatasetPolicy`` → ``DatasetLoader``)."""

    @abstractmethod
    def bind(self, num_fewshot: int) -> "FewShotGenerator":
        """Resolve the shot count — failing fast on an unsupported request, or pinning it for a fixed-shot
        policy — and produce the per-run generator that holds it. Called at eval creation, before any load."""

    @abstractmethod
    def documentation(self) -> FewShotDoc:
        """The demonstration split and example shot count for the rendered task docs."""


class FewShotGenerator(ABC):
    """A per-run few-shot worker, bound to a shot count: it remembers its demonstration pool once the data is
    loaded, then renders the demonstrations to show before each eval item."""

    @abstractmethod
    def prepare(self, dataset: Mapping[str, Any], *, sample_split: str, sample_rows: list[dict[str, Any]]) -> None:
        """Remember the demonstration pool from the already-loaded ``dataset`` (called once per subject during
        data loading). A same-split draw reuses the eval's already-shuffled ``sample_rows``, so demonstrations
        are never re-ordered relative to the eval items."""

    @abstractmethod
    def for_item(self, item: dict[str, Any], rnd: random.Random) -> list[FewshotExample]:
        """The rendered demonstrations to show before ``item`` — leak-safe against ``item`` itself."""

    @abstractmethod
    def metadata(self) -> dict[str, str]:
        """Few-shot metadata merged into the eval's ``get_metadata`` (e.g. the source split)."""


def draw_demonstrations(
    pool: list[dict[str, Any]],
    *,
    is_sample_split: bool,
    item: dict[str, Any],
    count: int,
    rnd: random.Random,
) -> list[dict[str, Any]]:
    """Draw ``count`` rows from ``pool``. When the pool is the sample split, over-sample by one and drop the
    current ``item`` if it was drawn — so its own answer never leaks — then truncate back; else draw directly.

    Shared by ``_SampledGenerator`` and by benchmarks whose demonstration *rendering* is item-dependent and so
    keep a local generator (e.g. Global-MMLU renders each shot in the current item's language)."""
    if count <= 0:
        return []
    if is_sample_split:
        drawn = rnd.sample(pool, count + 1)
        drawn = [row for row in drawn if row != item]
        return drawn[:count]
    return rnd.sample(pool, count)


@final
class SampledFewShot(FewShotPolicy):
    """Draws demonstrations at random from ``split`` (optionally restricted to rows passing ``keep``), each
    rendered by ``renderer``. When ``split`` is the sample split, the current eval item is excluded so its own
    answer never leaks into its prompt."""

    def __init__(
        self,
        split: str,
        renderer: FewShotRenderer,
        *,
        keep: Callable[[dict[str, Any]], bool] | None = None,
    ) -> None:
        self._split = split
        self._renderer = renderer
        self._keep = keep

    @override
    def bind(self, num_fewshot: int) -> FewShotGenerator:
        return _SampledGenerator(num_fewshot, split=self._split, keep=self._keep, renderer=self._renderer)

    @override
    def documentation(self) -> FewShotDoc:
        return FewShotDoc(split=self._split, example_shots=1)


@final
class _SampledGenerator(FewShotGenerator):
    def __init__(
        self,
        count: int,
        *,
        split: str,
        keep: Callable[[dict[str, Any]], bool] | None,
        renderer: FewShotRenderer,
    ) -> None:
        self._count = count
        self._split = split
        self._keep = keep
        self._renderer = renderer
        self._pool: list[dict[str, Any]] = []
        self._is_sample_split = False

    @override
    def prepare(self, dataset: Mapping[str, Any], *, sample_split: str, sample_rows: list[dict[str, Any]]) -> None:
        if self._count <= 0:
            return  # nothing will be drawn, so a separate few-shot split need not even be present
        self._is_sample_split = self._split == sample_split
        rows = sample_rows if self._is_sample_split else list(dataset[self._split])
        self._pool = [row for row in rows if self._keep(row)] if self._keep is not None else rows

    @override
    def for_item(self, item: dict[str, Any], rnd: random.Random) -> list[FewshotExample]:
        drawn = draw_demonstrations(
            self._pool, is_sample_split=self._is_sample_split, item=item, count=self._count, rnd=rnd
        )
        return [self._renderer.render(row) for row in drawn]

    @override
    def metadata(self) -> dict[str, str]:
        return {"fewshot_split": self._split}


@final
class PredefinedFewShot(FewShotPolicy):
    """A fixed, hand-written set of demonstrations (not drawn from the dataset), each rendered by ``renderer``.
    The shot count is pinned to ``count`` — a benchmark whose prompt uses a canonical fixed few-shot block —
    warning (rather than sampling differently) if a different count is requested."""

    def __init__(
        self,
        items: list[dict[str, Any]],
        renderer: FewShotRenderer,
        *,
        count: int,
        label: str,
    ) -> None:
        self._items = items
        self._renderer = renderer
        self._count = count
        self._label = label

    @override
    def bind(self, num_fewshot: int) -> FewShotGenerator:
        if num_fewshot != self._count:
            logger.warning(f"{self._label} uses a fixed num_fewshot of {self._count}. Got {num_fewshot}.")
        return _PredefinedGenerator(self._count, items=self._items, renderer=self._renderer)

    @override
    def documentation(self) -> FewShotDoc:
        return FewShotDoc(split=None, example_shots=self._count)


@final
class _PredefinedGenerator(FewShotGenerator):
    def __init__(self, count: int, *, items: list[dict[str, Any]], renderer: FewShotRenderer) -> None:
        self._count = count
        self._items = items
        self._renderer = renderer

    @override
    def prepare(self, dataset: Mapping[str, Any], *, sample_split: str, sample_rows: list[dict[str, Any]]) -> None:
        return None  # fixed exemplars, nothing to remember

    @override
    def for_item(self, item: dict[str, Any], rnd: random.Random) -> list[FewshotExample]:
        return [self._renderer.render(demo) for demo in self._items[: self._count]]

    @override
    def metadata(self) -> dict[str, str]:
        return {"fewshot_split": "predefined"}


@final
class NoFewShot(FewShotPolicy):
    """A benchmark that only runs 0-shot: it rejects any few-shot request and shows no demonstrations."""

    @override
    def bind(self, num_fewshot: int) -> FewShotGenerator:
        if num_fewshot != 0:
            raise ValueError(f"This benchmark is 0-shot only; num_fewshot must be 0, got {num_fewshot}.")
        return _NoGenerator()

    @override
    def documentation(self) -> FewShotDoc:
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
    def metadata(self) -> dict[str, str]:
        return {}
