from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from eval_framework.tasks.task import EvalFactory, ResponseType, Task
from template_formatting.formatter import BaseFormatter

if TYPE_CHECKING:
    from eval_framework.metrics.base import BaseMetric


class Lazy(EvalFactory):
    """An ``EvalFactory`` that defers producing the real factory until first use.

    Holds an ``id`` and a ``load`` thunk returning the wrapped ``EvalFactory``; every
    other call is delegated to that factory, which is built (and memoized) on first
    access. This keeps ``Lazy`` independent of any concrete task type — the thunk owns
    whatever loading (e.g. importing a module) the wrapped factory requires.
    """

    def __init__(self, id: str, load: Callable[[], EvalFactory]) -> None:
        self._id = id
        self._load = load
        self._factory: EvalFactory | None = None

    def _loaded_factory(self) -> EvalFactory:
        if self._factory is None:
            self._factory = self._load()
        return self._factory

    def id(self) -> str:
        return self._id

    def create(
        self,
        num_fewshot: int,
        custom_subjects: list[str] | None,
        custom_hf_revision: str | None,
        user_prompt_suffix: str | None = None,
        seed: int | None = None,
    ) -> Task:
        return self._loaded_factory().create(num_fewshot, custom_subjects, custom_hf_revision, user_prompt_suffix, seed)

    def response_type(self) -> ResponseType:
        return self._loaded_factory().response_type()

    def metrics(self) -> list[type["BaseMetric"]]:
        return self._loaded_factory().metrics()

    def subjects(self) -> list[Any]:
        return self._loaded_factory().subjects()

    def display_name(self) -> str:
        return self._loaded_factory().display_name()

    def markdown_doc(self, formatters: Sequence[BaseFormatter]) -> str:
        return self._loaded_factory().markdown_doc(formatters)
