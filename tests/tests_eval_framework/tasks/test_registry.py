import functools
from collections.abc import Callable

import pytest

from eval_framework.tasks.benchmarks.math_reasoning import (
    HendrycksMath_EleutherAI_EN,
    HendrycksMath_EleutherAI_EN_Level5,
)
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
    register_task(HendrycksMath_EleutherAI_EN)

    assert is_registered("HendrycksMath_EleutherAI_EN")
    assert set(registered_task_names()) == {"HendrycksMath_EleutherAI_EN"}
    assert get_task("HendrycksMath_EleutherAI_EN") is HendrycksMath_EleutherAI_EN
    assert get_task("hendrycksmath_eleutherai_en") is HendrycksMath_EleutherAI_EN
    assert get_task("HENDRYCKSMATH_ELEUTHERAI_EN") is HendrycksMath_EleutherAI_EN

    register_task(HendrycksMath_EleutherAI_EN_Level5)
    assert set(registered_task_names()) == {
        "HendrycksMath_EleutherAI_EN",
        "HendrycksMath_EleutherAI_EN_Level5",
    }
    assert get_task("hendrycksmath eleutherai en level5") is HendrycksMath_EleutherAI_EN_Level5
    assert get_task("HENDRYCKSMATH ELEUTHERAI EN LEVEL5") is HendrycksMath_EleutherAI_EN_Level5
    assert get_task("HendrycksMath_EleutherAI_EN_Level5") is HendrycksMath_EleutherAI_EN_Level5
    assert get_task("Hendrycks-Math_EleutherAI EN Level5") is HendrycksMath_EleutherAI_EN_Level5

    with pytest.raises(ValueError):
        get_task("HendrycksMath.EleutherAI.EN.Level5")


@temporary_registry
def test_register_non_task() -> None:
    with pytest.raises(ValueError):
        register_task(int)  # type: ignore[arg-type]

    class MyTask:
        pass

    with pytest.raises(ValueError):
        register_task(MyTask)  # type: ignore[arg-type]


@temporary_registry
def test_lazy_registration() -> None:
    register_lazy_task(f"{HendrycksMath_EleutherAI_EN.__module__}.{HendrycksMath_EleutherAI_EN.__name__}")
    assert get_task("hendrycksmath_eleutherai_en") is HendrycksMath_EleutherAI_EN
