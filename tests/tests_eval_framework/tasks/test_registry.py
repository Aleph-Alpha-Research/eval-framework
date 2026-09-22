import functools
from collections.abc import Callable, Iterable

import pytest

from eval_framework.tasks.base import BaseTask, Language, ResponseType, Sample
from eval_framework.tasks.registry import Registry, register_lazy_task, register_task, with_registry


class BenchmarkStub(BaseTask[str]):
    """A minimal task used only to exercise registry registration and lookup — its topic is irrelevant.
    ``NAME`` normalizes to the class name so eager and lazy registration key it identically."""

    REVISION_LOCKFILE = None

    NAME = "Benchmark Stub"
    DATASET_PATH = "stub"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "test"
    SUBJECTS = ["stub"]
    LANGUAGE = Language.ENG
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS: list = []

    def iterate_samples(self, num_samples: int | None = None) -> Iterable[Sample]:
        return iter([])


class MultiWordBenchmarkStub(BenchmarkStub):
    """A longer name, to exercise whitespace/case/punctuation normalization in lookups."""

    NAME = "Multi Word Benchmark Stub"


def temporary_registry[**P, T](fun: Callable[P, T]) -> Callable[P, T]:
    """Decorator to run a function with a temporary empty task registry."""

    @functools.wraps(fun)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        temp_registry = Registry()
        with with_registry(temp_registry):
            return fun(*args, **kwargs)

    return wrapper


def test_case_insensitive_lookup() -> None:
    registry = Registry()

    register_task(BenchmarkStub, registry)

    assert "Benchmark Stub" in registry
    assert set(registry.task_names()) == {"BenchmarkStub"}
    assert registry["Benchmark Stub"].id() == BenchmarkStub.__name__
    assert registry["benchmark stub"].id() == BenchmarkStub.__name__
    assert registry["BenchmarkStub"].id() == BenchmarkStub.__name__

    register_task(MultiWordBenchmarkStub, registry)
    assert set(registry.task_names()) == {"BenchmarkStub", "MultiWordBenchmarkStub"}
    assert registry["multi word benchmark stub"].id() == MultiWordBenchmarkStub.__name__
    assert registry["MULTI WORD BENCHMARK STUB"].id() == MultiWordBenchmarkStub.__name__
    assert registry["Multi Word     Benchmark Stub"].id() == MultiWordBenchmarkStub.__name__
    assert registry["Multi-Word_Benchmark-Stub"].id() == MultiWordBenchmarkStub.__name__

    with pytest.raises(ValueError):
        registry["Multi.Word.Benchmark.Stub"]


def test_register_non_task() -> None:
    registry = Registry()

    with pytest.raises(ValueError):
        register_task(int, registry=registry)  # type: ignore[arg-type]

    class MyTask:
        pass

    with pytest.raises(ValueError):
        register_task(MyTask, registry=registry)  # type: ignore[arg-type]


def test_lazy_registration() -> None:
    registry = Registry()
    register_lazy_task(f"{BenchmarkStub.__module__}.{BenchmarkStub.__name__}", registry=registry)
    assert registry["Benchmark Stub"].display_name() == BenchmarkStub.NAME


def test_subjects() -> None:
    registry = Registry()
    register_task(BenchmarkStub, registry)
    assert registry["Benchmark Stub"].subjects() == BenchmarkStub.SUBJECTS

    registry = Registry()
    register_lazy_task(f"{BenchmarkStub.__module__}.{BenchmarkStub.__name__}", registry=registry)
    assert registry["Benchmark Stub"].subjects() == BenchmarkStub.SUBJECTS


def test_deprecated_register_methods_warn() -> None:
    registry = Registry()
    with pytest.warns(DeprecationWarning):
        registry.register(BenchmarkStub)
    assert registry["Benchmark Stub"].id() == BenchmarkStub.__name__

    registry = Registry()
    with pytest.warns(DeprecationWarning):
        registry.register_lazy(f"{MultiWordBenchmarkStub.__module__}.{MultiWordBenchmarkStub.__name__}")
    assert registry["Multi Word Benchmark Stub"].subjects() == MultiWordBenchmarkStub.SUBJECTS
