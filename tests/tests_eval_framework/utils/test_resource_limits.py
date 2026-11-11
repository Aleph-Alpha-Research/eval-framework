import resource
import signal
import sys
import threading
import time
from types import FrameType

import numpy as np
import pytest

from eval_framework.utils.resource_limits import memory_limited, time_limited

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


linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="memory_limited only caps on Linux")


# The limit caps new address space, not allocation size: glibc may serve blocks below its mmap threshold
# (raised dynamically up to 32 MiB) from memory the process already holds. Above it every allocation is a
# fresh mapping, so the boundary is exact (64 MiB less 24 bytes of allocator overhead) whatever ran before.
@memory_limited(64 * 2**20)
def _allocate_mib(mib: int) -> int:
    return np.ones(mib * 2**20, dtype=np.uint8).nbytes


@linux_only
def test_memory_limited_allows_an_allocation_just_within_the_headroom() -> None:
    assert _allocate_mib(63) == 63 * 2**20


@linux_only
def test_memory_limited_refuses_an_allocation_just_beyond_the_headroom() -> None:
    with pytest.raises(MemoryError):
        _allocate_mib(65)


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
