import contextlib
import importlib
import re
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterator, Sequence
from typing import Any

from eval_framework.tasks.base import BaseTask
from eval_framework.utils.packaging import is_extra_installed, validate_package_extras

__all__ = [
    "register_task",
    "register_lazy_task",
    "BenchmarkFactory",
    "Registry",
    "with_registry",
    "get_task",
    "registered_tasks_iter",
    "is_registered",
    "validate_task_name",
    "registered_task_names",
]


class BenchmarkFactory(ABC):
    """Produces a registered benchmark's task.

    The registry stores one factory per benchmark. This allows the factory to be
    constructed without constructing all benchmarks. Going via this ABC allows
    the factory instances to contain state specifically relevant to the
    benchmark, as well as supporting different strategies for instantiating it.
    E.g. eager vs lazy loading of the required dependencies.
    """

    @abstractmethod
    def task_class(self) -> type[BaseTask]:
        """Return the task class, importing it on first access if necessary."""

    @property
    @abstractmethod
    def source_module(self) -> str:
        """Module the task class is defined in, resolvable without importing it."""


class _Lazy(BenchmarkFactory):
    """
    Create benchmark from qualified class path; Delays importing modules until
    benchmark is constructed.
    """

    def __init__(self, class_name: str, module: str, extras: Sequence[str] = ()) -> None:
        """
        Args:
            class_name: The name of the task class to import.
            module: The module to import the task class from.
            extras: Extra dependencies of `eval_framework` required for this task.
        """
        self._class_name = class_name
        self._module = module
        self._extras = tuple(validate_package_extras(extras))
        self._loaded: type[BaseTask] | None = None

    @property
    def source_module(self) -> str:
        return self._module

    def task_class(self) -> type[BaseTask]:
        if self._loaded is None:
            for extra in self._extras:
                if not is_extra_installed(extra):
                    raise ImportError(f"The required package eval_framework[{extra}] is not installed.")
            module = importlib.import_module(self._module)
            self._loaded = getattr(module, self._class_name)
        return self._loaded


class _Eager(BenchmarkFactory):
    """Wraps an already-imported task class."""

    def __init__(self, task: type[BaseTask]) -> None:
        self._task = task

    @property
    def source_module(self) -> str:
        return self._task.__module__

    def task_class(self) -> type[BaseTask]:
        return self._task


class Registry:
    """A registry for tasks with support for lazy loading.

    Task names are hashed based on the upper-case name, to avoid issues with
    ambiguous naming.
    """

    def __init__(self) -> None:
        # TODO: Lookup only with upper names
        self._registry: dict[str, tuple[str, BenchmarkFactory]] = dict()

    def __iter__(self) -> Iterator[str]:
        for name, _ in self._registry.values():
            yield name

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

    def __getitem__(self, name: str, /) -> type[BaseTask]:
        task_key = self._task_key(name)
        try:
            _, factory = self._registry[task_key]
        except KeyError:
            raise KeyError(f"Task not found: {name}")

        return factory.task_class()

    def add(self, task: type[BaseTask]) -> None:
        task_key = self._task_key(task.NAME)
        self._registry[task_key] = (task.NAME, _Eager(task))

    def __setitem__(self, name: str, factory: BenchmarkFactory) -> None:
        task_key = self._task_key(name)
        if task_key in self._registry:
            raise ValueError(f"Cannot register duplicate task with key: {task_key}")

        self._registry[task_key] = (name, factory)


_REGISTRY = Registry()


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
    if not is_registered(name):
        raise ValueError(f"Task not registered: {name}")
    return name


def registered_tasks_iter() -> Iterator[tuple[str, type[BaseTask]]]:
    """Iterate over the names and classes of all registered tasks.

    Note: This method will import any lazily registered task.
    """
    for name in registered_task_names():
        yield name, get_task(name)


def get_task(name: str, /) -> type[BaseTask]:
    """Return a registered task for a given name.

    Note: This method will import any lazily registered task.
    """
    return _REGISTRY[name]


def register_task(task: type[BaseTask]) -> str:
    """The class name is used as the task name."""
    if not issubclass(task, BaseTask):
        raise ValueError(f"Can only register subclasses of BaseTask, got {task}")
    name = task.__name__
    _REGISTRY[name] = _Eager(task)
    return name


def register_lazy_task(class_path: str, /, *, extras: Sequence[str] = ()) -> None:
    """Register a task without importing it.

    Lazily register a task without importing the module.

    Args:
        class_path: The full path to the task class. For example,
            `eval_framework.tasks.benchmarks.truthfulqa.TRUTHFULQA`.
        extras: Any extra dependencies of `eval_framework` that need to be installed for this task.
    """
    if isinstance(extras, str):
        extras = [extras]
    if "." not in class_path:
        raise ValueError(
            f"Invalid class path `{class_path}`. This needs to be a global path like "
            "`eval_framework.tasks.benchmarks.truthfulqa.TRUTHFULQA`): "
        )

    base_module, class_name = class_path.rsplit(".", maxsplit=1)
    _REGISTRY[class_name] = _Lazy(class_name=class_name, module=base_module, extras=extras)
