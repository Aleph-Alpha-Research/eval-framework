"""Decorators that bound the resources a single call may use."""

import functools
import signal
import sys
import threading
from collections.abc import Callable
from types import FrameType
from typing import ParamSpec, TypeVar

if sys.platform == "linux":
    import resource

P = ParamSpec("P")
R = TypeVar("R")


def time_limited(seconds: int) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Raise ``TimeoutError`` from the decorated call once it has run for ``seconds``.

    Uses SIGALRM: Unix only, callable only from the main thread (``signal.signal`` raises ``ValueError``
    elsewhere), and a single process-wide timer, so it raises ``RuntimeError`` rather than cancel an alarm
    that is already pending (e.g. when nested). Long-running native code is only interrupted once it returns
    to the interpreter.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            def _raise_timeout(signum: int, frame: FrameType | None) -> None:
                raise TimeoutError(f"{func.__qualname__} exceeded {seconds}s")

            if signal.getitimer(signal.ITIMER_REAL)[0] > 0:
                # One process-wide timer: arming ours would silently cancel the pending one.
                raise RuntimeError(f"{func.__qualname__} cannot run while another alarm is pending")
            if signal.getsignal(signal.SIGALRM) is None:
                # Installed before Python started (e.g. when embedded); it could not be reinstated afterwards.
                raise RuntimeError(f"{func.__qualname__} would replace a SIGALRM handler installed outside Python")
            previous_handler = signal.signal(signal.SIGALRM, _raise_timeout)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous_handler)

        return wrapper

    return decorator


def _address_space_bytes() -> int:
    with open("/proc/self/statm") as statm:
        return int(statm.read().split()[0]) * resource.getpagesize()


def memory_limited(headroom_bytes: int) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Let the decorated call grow the process's address space by at most ``headroom_bytes``.

    A larger allocation raises ``MemoryError`` instead of succeeding under memory overcommit and getting
    the process OOM-killed while it fills the pages. Lowers the soft ``RLIMIT_AS`` of this process (not
    the node or container) for the duration of the call, so it caps every thread meanwhile; callable only
    from the main thread (raises ``RuntimeError`` elsewhere). Linux only; elsewhere the call runs uncapped.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if sys.platform != "linux":
                return func(*args, **kwargs)
            if threading.current_thread() is not threading.main_thread():
                # Concurrent calls would restore each other's lowered limits.
                raise RuntimeError(f"{func.__qualname__} is memory-limited and can only run on the main thread")
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            limit = _address_space_bytes() + headroom_bytes
            # Never loosen a limit that is already in place, e.g. from an enclosing call.
            limit = min(x for x in (limit, soft, hard) if x != resource.RLIM_INFINITY)
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
            try:
                return func(*args, **kwargs)
            finally:
                resource.setrlimit(resource.RLIMIT_AS, (soft, hard))

        return wrapper

    return decorator
