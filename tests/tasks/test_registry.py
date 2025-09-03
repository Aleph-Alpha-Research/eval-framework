import functools
from collections.abc import Callable

from eval_framework.tasks.benchmarks.math_reasoning import MATH, MATHLvl5
from eval_framework.tasks.registry import (
    Registry,
    get_task,
    is_registered,
    register_lazy_task,
    register_task,
    registered_task_names,
    with_registry,
)


def temporary_registry[**P, T](fun: Callable[P, T]) -> Callable[P, T]:
    """Decorator to run a function with a temporary empty task registry."""

    @functools.wraps(fun)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        temp_registry = Registry()
        with with_registry(temp_registry):
            return fun(*args, **kwargs)

    return wrapper


@temporary_registry
def test_case_insensitive_lookup() -> None:
    register_task("MATH", MATH)

    assert is_registered("MATH")
    assert set(registered_task_names()) == {"MATH"}
    assert get_task("MATH") is MATH
    assert get_task("Math") is MATH
    assert get_task("math") is MATH

    register_task("MATH Lvl 5", MATHLvl5)
    assert set(registered_task_names()) == {"MATH", "MATH Lvl 5"}
    assert get_task("math lvl 5") is MATHLvl5
    assert get_task("MATH LVL 5") is MATHLvl5
    assert get_task("Math Lvl 5") is MATHLvl5


@temporary_registry
def test_lazy_registration() -> None:
    register_lazy_task("whatever", class_path=f"{MATH.__module__}.{MATH.__name__}")
    assert get_task("whatever") is MATH
