import functools
from collections.abc import Callable

import pytest

from eval_framework.tasks.benchmarks.math_reasoning import MATH, MATHLvl5
from eval_framework.tasks.registry import Registry, with_registry


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

    registry.register(MATH)

    assert "MATH" in registry
    assert set(registry.task_names()) == {"MATH"}
    assert registry["MATH"].id() == MATH.__name__
    assert registry["Math"].id() == MATH.__name__
    assert registry["math"].id() == MATH.__name__

    registry.register(MATHLvl5)
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
        registry.register(int)  # type: ignore[arg-type]

    class MyTask:
        pass

    with pytest.raises(ValueError):
        registry.register(MyTask)  # type: ignore[arg-type]


def test_lazy_registration() -> None:
    registry = Registry()
    registry.register_lazy(f"{MATH.__module__}.{MATH.__name__}")
    assert registry["Math"].display_name() == MATH.NAME
