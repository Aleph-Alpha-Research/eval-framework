import contextlib
import importlib
import re
import warnings
from collections.abc import Generator, Iterator
from typing import Any

from eval_framework.tasks.base import BaseTask, Eager
from eval_framework.tasks.lazy import Lazy
from eval_framework.tasks.task import EvalFactory

__all__ = [
    "register_task",
    "register_lazy_task",
    "EvalFactory",
    "Eager",
    "Lazy",
    "Registry",
    "with_registry",
    "is_registered",
    "validate_task_name",
    "registered_task_names",
]


class Registry:
    """A registry for Tasks"""

    def __init__(self) -> None:
        self._registry: dict[str, EvalFactory] = dict()

    def __iter__(self) -> Iterator[str]:
        """Iterate over all task names in the registry."""
        for factory in self._registry.values():
            yield factory.id()

    def task_names(self) -> list[str]:
        """The names of all registered tasks."""
        return list(self)

    def items(self) -> Iterator[tuple[str, EvalFactory]]:
        """Iterate over `(task name, EvalFactory)` pairs in the registry."""
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

    def __getitem__(self, name: str, /) -> EvalFactory:
        task_key = self._task_key(name)
        try:
            return self._registry[task_key]
        except KeyError:
            raise KeyError(f"Task not found: {name=} with task_key {task_key=}")

    def add(self, factory: EvalFactory) -> None:
        """Register a factory under the key derived from its ``id()``."""
        task_key = self._task_key(factory.id())
        if task_key in self._registry:
            raise ValueError(f"Cannot register duplicate task with key: {task_key}")

        self._registry[task_key] = factory

    def register(self, task: type[BaseTask]) -> str:
        """Register a task class. The class name is used as the task name.

        .. deprecated::
            Use :func:`register_task` (``register_task(task, registry)``) instead.
        """
        warnings.warn(
            "Registry.register is deprecated; use register_task(task, registry) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return register_task(task, registry=self)

    def register_lazy(self, class_path: str, /) -> None:
        """Register a task by its dotted class path, without importing its module.

        .. deprecated::
            Use :func:`register_lazy_task` (``register_lazy_task(class_path, registry)``) instead.
        """
        warnings.warn(
            "Registry.register_lazy is deprecated; use register_lazy_task(class_path, registry) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        register_lazy_task(class_path, registry=self)


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


def register_task(task: type[BaseTask], registry: Registry | None = None) -> str:
    """Register ``task`` into ``registry`` (the global registry by default).

    The class name is used as the task name.
    """
    if not issubclass(task, BaseTask):
        raise ValueError(f"Can only register subclasses of BaseTask, got {task}")
    r = registry if registry is not None else _REGISTRY
    factory = Eager(task)
    r.add(factory)
    return factory.id()


def register_lazy_task(class_path: str, /, registry: Registry | None = None) -> None:
    """Register a task by its dotted class path, without importing its module."""
    if "." not in class_path:
        raise ValueError(
            f"Invalid class path `{class_path}`. This needs to be a global path like "
            "`eval_framework.tasks.benchmarks.mmlu.MMLU`): "
        )
    r = registry if registry is not None else _REGISTRY
    module_path, class_name = class_path.rsplit(".", maxsplit=1)

    def load() -> EvalFactory:
        module = importlib.import_module(module_path)
        return Eager(getattr(module, class_name))

    r.add(Lazy(id=class_name, load=load))
