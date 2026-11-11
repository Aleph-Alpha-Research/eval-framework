"""Decorators that bound the resources a single call may use."""

import functools
import signal
from collections.abc import Callable
from types import FrameType
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def time_limited(seconds: int) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Raise ``TimeoutError`` from the decorated call once it has run for ``seconds``.

    Uses SIGALRM: Unix only, callable only from the main thread (``signal.signal`` raises ``ValueError``
    elsewhere), and a single process-wide timer, so limited calls must not nest. Long-running native code
    is only interrupted once it returns to the interpreter.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            def _raise_timeout(signum: int, frame: FrameType | None) -> None:
                raise TimeoutError(f"{func.__qualname__} exceeded {seconds}s")

            previous_handler = signal.signal(signal.SIGALRM, _raise_timeout)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous_handler)

        return wrapper

    return decorator
