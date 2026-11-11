import signal
import sys
import time
from types import FrameType

import pytest

from eval_framework.utils.resource_limits import time_limited

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
