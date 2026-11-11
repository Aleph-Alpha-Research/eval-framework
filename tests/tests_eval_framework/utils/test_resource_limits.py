import gc
import signal
import sys
import threading
import time
from types import FrameType

import numpy as np
import pytest

from eval_framework.utils.resource_limits import memory_limited, time_limited

if sys.platform != "win32":
    import resource  # used by the Linux-only tests; absent on Windows

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="time_limited relies on SIGALRM")


def test_time_limited_interrupts_a_slow_call() -> None:
    @time_limited(1)
    def slow() -> None:
        time.sleep(5)

    start = time.monotonic()
    with pytest.raises(TimeoutError):
        slow()

    assert time.monotonic() - start < 3


def test_time_limited_returns_the_result_of_a_fast_call_and_leaves_no_alarm() -> None:
    @time_limited(1)
    def add_one(x: int) -> int:
        return x + 1

    assert add_one(2) == 3
    assert signal.alarm(0) == 0  # returns the seconds left on any pending alarm


def test_time_limited_restores_the_previous_sigalrm_handler() -> None:
    def previous_handler(signum: int, frame: FrameType | None) -> None:
        pass

    original = signal.signal(signal.SIGALRM, previous_handler)
    try:
        time_limited(1)(lambda: None)()

        assert signal.getsignal(signal.SIGALRM) is previous_handler
    finally:
        signal.signal(signal.SIGALRM, original)


def test_time_limited_refuses_to_replace_a_handler_installed_outside_python(monkeypatch: pytest.MonkeyPatch) -> None:
    # Python reports a handler that was in place before it started (e.g. when embedded) as None, and could
    # not reinstate it once replaced.
    real_handler = signal.getsignal(signal.SIGALRM)
    monkeypatch.setattr(signal, "getsignal", lambda signum: None)

    with pytest.raises(RuntimeError):
        time_limited(1)(lambda: None)()

    monkeypatch.undo()
    assert signal.getsignal(signal.SIGALRM) is real_handler


def test_time_limited_refuses_to_cancel_a_pending_alarm() -> None:
    signal.alarm(100)  # an outer timer, e.g. an enclosing time_limited call
    try:
        with pytest.raises(RuntimeError):
            time_limited(1)(lambda: None)()

        assert signal.alarm(0) > 0  # still pending: returns the seconds left
    finally:
        signal.alarm(0)


linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="memory_limited only caps on Linux")


# The limit caps the net growth of the address space, not the allocation size:
# - glibc may serve blocks below its mmap threshold (raised dynamically up to 32 MiB) from memory the process
#   already holds, so both sizes below stay above it;
# - memory freed during the call (e.g. CPython releasing an empty 1 MiB arena) makes room, and memory
#   allocated meanwhile takes it. In a pytest worker that has run other tests this shifts the boundary by a
#   few MiB, so both sizes keep 16 MiB clear of the headroom, and earlier tests' garbage is collected first
#   so a collection pass can't free a large block mid-call.
@memory_limited(64 * 2**20)
def _allocate_mib_capped(mib: int) -> int:
    return np.ones(mib * 2**20, dtype=np.uint8).nbytes


def _allocate_mib(mib: int) -> int:
    gc.collect()
    return _allocate_mib_capped(mib)


@linux_only
def test_memory_limited_allows_an_allocation_within_the_headroom() -> None:
    assert _allocate_mib(48) == 48 * 2**20


@linux_only
def test_memory_limited_refuses_an_allocation_beyond_the_headroom() -> None:
    with pytest.raises(MemoryError):
        _allocate_mib(80)


@linux_only
def test_memory_limited_restores_the_previous_limit() -> None:
    before = resource.getrlimit(resource.RLIMIT_AS)

    @memory_limited(64 * 2**20)
    def allocate_1_mib() -> int:
        return len(bytearray(2**20))

    assert allocate_1_mib() == 2**20
    assert resource.getrlimit(resource.RLIMIT_AS) == before


@linux_only
def test_memory_limited_refuses_to_run_off_the_main_thread() -> None:
    # The limit is process-wide: concurrent calls would restore each other's lowered limits.
    @memory_limited(64 * 2**20)
    def noop() -> None:
        pass

    errors: list[BaseException] = []

    def call() -> None:
        try:
            noop()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=call)
    thread.start()
    thread.join()

    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
