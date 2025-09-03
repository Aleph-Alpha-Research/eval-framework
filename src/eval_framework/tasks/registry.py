import contextlib
import importlib.metadata
import importlib.util
from collections.abc import Generator, Iterator, Sequence
from typing import Annotated, Any

import pydantic
from packaging.requirements import Requirement
from packaging.version import Version
from pydantic import AfterValidator

from eval_framework.tasks.base import BaseTask

__all__ = [
    "register_task",
    "register_lazy_task",
    "Registry",
    "with_registry",
    "get_task",
    "registered_tasks_iter",
    "is_registered",
    "validate_task_name",
    "registered_task_names",
]


def _validate_package_extras(extras: str | Sequence[str], /, *, package: str = "eval_framework") -> Sequence[str]:
    """Validate that the specified extras are valid for the given package."""
    if isinstance(extras, str):
        extras = [extras]

    metadata = importlib.metadata.metadata(package)
    package_extras = set(metadata.get_all("Provides-Extra") or [])
    for extra in extras:
        if extra not in package_extras:
            raise ValueError(f"Invalid extra: {extra}. Options are {package_extras}")

    return extras


def validate_import_path(import_path: str) -> str:
    if importlib.util.find_spec(import_path) is None:
        raise ValueError(f"Invalid import path: {import_path}")
    return import_path


def extra_requires(extra: str, /, *, package: str = "eval_framework") -> list[str]:
    _validate_package_extras(extra, package=package)
    dist = importlib.metadata.distribution(package)
    requires = dist.requires or []
    extra_str = f"extra == '{extra}'"
    return [r.split(";")[0].strip() for r in requires if r.endswith(extra_str)]


def _dependency_satisfied(dep: str, /) -> bool:
    """Return True if the dependency string is satisfied.

    Args:
        A dependency string: for example "torch~=2.0".
    """
    try:
        dist = importlib.metadata.distribution(Requirement(dep).name)
        installed_version = Version(dist.version)
        req = Requirement(dep)
        return installed_version in req.specifier
    except (importlib.metadata.PackageNotFoundError, Exception):
        return False


def is_extra_installed(extra: str, package: str = "eval_framework") -> bool:
    """Return `True` if"""
    for req in extra_requires(extra, package=package):
        if not _dependency_satisfied(req):
            return False
    return True


class TaskPlaceholder(pydantic.BaseModel, extra="forbid", frozen=True):
    name: Annotated[
        str,
        "The name of the Task class that we want to import",
    ]
    module: Annotated[
        str,
        "The module from where to import the task",
        validate_import_path,
    ]
    extras: Annotated[
        tuple[str, ...],
        "Extra dependencies that are required for the task",
        AfterValidator(_validate_package_extras),
    ] = ()

    def load(self) -> type[BaseTask]:
        for extra in self.extras:
            if not is_extra_installed(extra):
                raise ImportError(f"The required package eval_framework[{extra}] is not installed.")
        module = importlib.import_module(self.module)
        return getattr(module, self.name)


class Registry:
    """A registry for tasks with support for lazy loading.

    Task names are hashed based on the upper-case name, to avoid issues with
    ambiguous naming.
    """

    def __init__(self) -> None:
        # TODO: Lookup only with upper names
        self._registry: dict[str, tuple[str, type[BaseTask] | TaskPlaceholder]] = dict()

    def __iter__(self) -> Iterator[str]:
        for name, _ in self._registry.values():
            yield name

    def __contains__(self, name: str) -> bool:
        task_key = name.upper()
        return task_key in self._registry

    def __getitem__(self, name: str, /) -> type[BaseTask]:
        task_key = name.upper()
        try:
            name, task = self._registry[task_key]
        except KeyError:
            raise KeyError(f"Task not found: {name}")

        if isinstance(task, TaskPlaceholder):
            task = task.load()
            self._registry[task_key] = (name, task)
        return task

    def add(self, task: type[BaseTask]) -> None:
        task_key = task.NAME.upper()
        self._registry[task_key] = (task.NAME, task)

    def __setitem__(self, name: str, task: type[BaseTask] | TaskPlaceholder) -> None:
        task_key = name.upper()
        if task_key in self._registry:
            raise ValueError(f"Cannot register duplicate task with key: {task_key}")

        self._registry[task_key] = (name, task)


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


def register_task(name: str, task: type[BaseTask]) -> None:
    _REGISTRY[name] = task


def register_lazy_task(name: str, class_path: str, *, extras: Sequence[str] = ()) -> None:
    """Register a task without importing it.

    Lazily register a task without importing the module.

    Args:
        name: The name of the task to register.
        class_path: The full path to the task class. For example,
            `eval_framework.tasks.benchmarks.truthfulqa.TRUTHFULQA`.
        extras: Any extra dependencies of `eval_framework` that need to be installed for this task.
    """
    if isinstance(extras, str):
        extras = [extras]

    base_module, class_name = class_path.rsplit(".", maxsplit=1)
    placeholder = TaskPlaceholder(name=class_name, module=base_module, extras=extras)
    _REGISTRY[name] = placeholder
