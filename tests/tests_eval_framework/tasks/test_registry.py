import functools
from collections.abc import Callable

import pytest

from eval_framework.tasks.benchmarks.math_reasoning import MATH, MATHLvl5
from eval_framework.tasks.registry import Registry, register_lazy_task, register_task, with_registry


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

    register_task(MATH, registry)

    assert "MATH" in registry
    assert set(registry.task_names()) == {"MATH"}
    assert registry["MATH"].id() == MATH.__name__
    assert registry["Math"].id() == MATH.__name__
    assert registry["math"].id() == MATH.__name__

    register_task(MATHLvl5, registry)
    assert set(registry.task_names()) == {"MATH", "MATHLvl5"}
    assert registry["math lvl 5"].id() == MATHLvl5.__name__
    assert registry["MATH LVL 5"].id() == MATHLvl5.__name__
    assert registry["Math Lvl 5"].id() == MATHLvl5.__name__
    assert registry["Math Lvl     5"].id() == MATHLvl5.__name__
    assert registry["Math-Lvl_5"].id() == MATHLvl5.__name__

    with pytest.raises(ValueError):
        registry["Math.Lvl.5"]


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
    register_lazy_task(f"{MATH.__module__}.{MATH.__name__}", registry=registry)
    assert registry["Math"].display_name() == MATH.NAME


def test_subjects() -> None:
    registry = Registry()
    register_task(MATH, registry)
    assert registry["MATH"].subjects() == MATH.SUBJECTS

    registry = Registry()
    register_lazy_task(f"{MATH.__module__}.{MATH.__name__}", registry=registry)
    assert registry["Math"].subjects() == MATH.SUBJECTS


def test_deprecated_register_methods_warn() -> None:
    registry = Registry()
    with pytest.warns(DeprecationWarning):
        registry.register(MATH)
    assert registry["MATH"].id() == MATH.__name__

    registry = Registry()
    with pytest.warns(DeprecationWarning):
        registry.register_lazy(f"{MATHLvl5.__module__}.{MATHLvl5.__name__}")
    assert registry["MATHLvl5"].subjects() == MATHLvl5.SUBJECTS
