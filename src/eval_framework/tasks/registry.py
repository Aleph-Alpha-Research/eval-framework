import contextlib
import re
from collections.abc import Generator, Iterator
from typing import Any

from eval_framework.contract import Benchmark
from eval_framework.tasks.lazy import Lazy

__all__ = [
    "Benchmark",
    "Lazy",
    "Registry",
    "with_registry",
    "is_registered",
    "validate_task_name",
    "registered_task_names",
]


class Registry:
    """A registry for Benchmarks"""

    def __init__(self) -> None:
        self._registry: dict[str, Benchmark] = dict()

    def __iter__(self) -> Iterator[str]:
        """Iterate over all task names in the registry."""
        for factory in self._registry.values():
            yield factory.id()

    def task_names(self) -> list[str]:
        """The names of all registered tasks."""
        return list(self)

    def items(self) -> Iterator[tuple[str, Benchmark]]:
        """Iterate over `(task name, Benchmark)` pairs in the registry."""
        for factory in self._registry.values():
            yield factory.id(), factory

    @staticmethod
    def _task_key(name: str, /) -> str:
        name = re.sub(r"[\s\-_]+", "", name).upper()
        if not name.isalnum():
            raise ValueError(
                f"Task name '{name}' contains invalid characters. Only alphanumeric characters are allowed."
            )
        return name

    def __contains__(self, name: str) -> bool:
        task_key = self._task_key(name)
        return task_key in self._registry

    def __getitem__(self, name: str, /) -> Benchmark:
        task_key = self._task_key(name)
        try:
            return self._registry[task_key]
        except KeyError:
            raise KeyError(f"Task not found: {name=} with task_key {task_key=}")

    def add(self, factory: Benchmark) -> None:
        """Register a factory under the key derived from its ``id()``."""
        task_key = self._task_key(factory.id())
        if task_key in self._registry:
            raise ValueError(f"Cannot register duplicate task with key: {task_key}")

        self._registry[task_key] = factory


_REGISTRY = Registry()


def registry() -> Registry:
    return _REGISTRY


@contextlib.contextmanager
def with_registry(registry: Registry) -> Generator[None, Any, None]:
    """Contextmanager to change the current registry."""
    global _REGISTRY
    old_registry = _REGISTRY
    try:
        _REGISTRY = registry
        yield
    finally:
        _REGISTRY = old_registry


def registered_task_names() -> list[str]:
    """Return the names of all registered tasks."""
    return list(_REGISTRY)


def is_registered(name: str, /) -> bool:
    """Return True if a task is registered."""
    return name in _REGISTRY


def validate_task_name(name: str) -> str:
    """Pydantic-style validator for task names."""
    if name not in registry():
        raise ValueError(f"Task not registered: {name}")
    return name
