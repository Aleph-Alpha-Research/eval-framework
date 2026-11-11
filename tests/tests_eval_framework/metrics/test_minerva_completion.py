import logging
import mmap
import subprocess
import sys
import time
from pathlib import Path

import pytest
from sympy.external.gmpy import GROUND_TYPES

from eval_framework.metrics.completion import minerva_math_utils
from eval_framework.metrics.completion.math_minerva_completion import MathMinervaCompletion
from eval_framework.metrics.completion.minerva_math_utils import extract_answers, is_equiv_minerva
from eval_framework.shared.types import Completion


def make_completion(completion: str, ground_truth: str | list[str]) -> Completion:
    """Helper to create Completion objects for testing."""
    return Completion(
        id=1,
        subject="test",
        ground_truth=ground_truth,
        prompt="test prompt",
        prompt_num_tokens=None,
        messages=None,
        completion=completion,
        raw_completion=completion,
        raw_completion_num_tokens=None,
    )


def _resident_bytes(pid: int) -> int:
    # /proc/<pid>/statm: sizes in pages, "total resident shared text lib data dirty"
    _total, resident, *_ = Path(f"/proc/{pid}/statm").read_text().split()
    return int(resident) * mmap.PAGESIZE


def _run_with_memory_growth_ceiling(setup: str, run: str, ceiling_bytes: int, timeout_s: float) -> str:
    """Run ``setup`` then ``run`` in a fresh interpreter and return what ``run`` prints.

    Kills the child as soon as its resident memory has grown more than ``ceiling_bytes`` beyond what it
    held after ``setup`` (the imports alone take a few hundred MB), and fails the test then or on timeout.
    Polls rather than capping the child, since a cap would itself turn the allocation under test into a
    MemoryError.
    """
    # The child waits after setup until the baseline is taken, so the run can't inflate it.
    code = f"{setup}\nimport sys\nprint('ready', flush=True)\nsys.stdin.readline()\n{run}"
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdin is not None and proc.stdout is not None
    failure = None
    try:
        if proc.stdout.readline() != "ready\n":
            failure = "setup failed"
        else:
            baseline = _resident_bytes(proc.pid)
            proc.stdin.write("go\n")
            proc.stdin.flush()
            deadline = time.monotonic() + timeout_s
            while failure is None and proc.poll() is None:
                grown = _resident_bytes(proc.pid) - baseline
                if grown > ceiling_bytes:
                    failure = f"killed after growing {grown / 2**20:.0f} MiB (ceiling {ceiling_bytes / 2**20:.0f} MiB)"
                elif time.monotonic() > deadline:
                    failure = f"killed after {timeout_s}s"
                time.sleep(0.005)
    finally:
        proc.kill()  # no-op once it has exited
        stdout, stderr = proc.communicate()
    if failure is not None:
        pytest.fail(f"{failure}; stderr tail:\n{stderr[-2000:]}")
    assert proc.returncode == 0, f"exit code {proc.returncode}; stderr tail:\n{stderr[-2000:]}"
    return stdout


class TestMathMinervaCompletion:
    """Tests for Minerva final-answer extraction."""

    def test_extract_answers_defaults_to_english(self) -> None:
        english = "Final Answer: The final answer is 24. I hope it is correct."
        german = "Finale Antwort: Die finale Antwort lautet 24. Ich hoffe, die Antwort ist korrekt."

        assert extract_answers(english)[0] == "24"
        assert extract_answers(german)[0] != "24"

    def test_extract_answers_can_select_german(self) -> None:
        german = "Finale Antwort: Die finale Antwort lautet 24. Ich hoffe, die Antwort ist korrekt."

        assert extract_answers(german, cot_style="minerva_de")[0] == "24"

    def test_extract_answers_german_relaxed_selection(self) -> None:
        german = "Endgültige Antwort: die Antwort ist 24 ich hoffe, sie ist korrekt."

        assert extract_answers(german, cot_style="minerva_de", relaxed=True)[0] == "24"
        assert extract_answers(german, relaxed=True)[0] != "24"

    def test_extract_answers_can_select_french(self) -> None:
        # Bare number: the ``$…$`` fallback of ``extract_answers`` must not mask a non-matching phrase.
        french = "Réponse finale : La réponse finale est 24. J'espère que la réponse est correcte."

        assert extract_answers(french, cot_style="minerva_fr")[0] == "24"
        assert extract_answers(french)[0] != "24"
        assert extract_answers(french, cot_style="minerva_de")[0] != "24"

    def test_extract_answers_french_strict_requires_canonical_wording(self) -> None:
        no_space_before_colon = "Réponse finale: La réponse finale est 24. J'espère que la réponse est correcte."

        assert extract_answers(no_space_before_colon, cot_style="minerva_fr")[0] != "24"
        assert extract_answers(no_space_before_colon, cot_style="minerva_fr", relaxed=True)[0] == "24"

    @pytest.mark.parametrize(
        "french",
        [
            "réponse finale : la réponse est 24. j'espère que c'est correct.",
            "Reponse finale : 24 J’espère qu’elle est correcte",
            "La réponse finale : La réponse finale est 24. J'espère que ce soit correct.",
        ],
    )
    def test_extract_answers_french_relaxed_selection(self, french: str) -> None:
        assert extract_answers(french, cot_style="minerva_fr", relaxed=True)[0] == "24"
        assert extract_answers(french, relaxed=True)[0] != "24"

    def test_is_equiv_minerva_refuses_implausibly_long_candidates_without_running_sympy(self) -> None:
        # Unguarded, parsing this burns the full 5s timeout.
        babble = "1.111011111011101111101111011011111111111011111111111.11111011" * 16

        start = time.perf_counter()
        result = is_equiv_minerva(babble, "\\sqrt{241}")

        assert result is False
        assert time.perf_counter() - start < 2.0

    def test_is_equiv_minerva_still_checks_long_strings_of_similar_length(self) -> None:
        long_expr = "+".join(["\\frac{1}{2}"] * 15)

        assert is_equiv_minerva(long_expr, long_expr) is True

    def test_is_equiv_minerva_logs_a_failed_check(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=minerva_math_utils.__name__):
            # An equation can't be subtracted from a number.
            assert is_equiv_minerva("x=3", "3") is False

        assert len(caplog.records) == 1
        assert "TypeError" in caplog.records[0].getMessage()

    @pytest.mark.skipif(not sys.platform.startswith("linux"), reason="measures memory via /proc")
    def test_is_equiv_minerva_stays_within_memory_on_power_towers(self) -> None:
        # SymPy evaluates integer powers eagerly: 2^{325^5} alone is ~450 GB of digits. With memory
        # overcommit (as on the eval pods) the allocation succeeds and the process is OOM-killed
        # while filling it, taking the whole eval run with it.
        # The child imports only the module under test (~90 MB). The warm-up check loads what simplify imports
        # lazily (numpy's OpenBLAS reserves per-core buffers), as thousands of checks would in a real run, so
        # the check under test grows the child by well under 1 MB.
        setup = (
            "from eval_framework.metrics.completion.minerva_math_utils import is_equiv_minerva\n"
            "is_equiv_minerva('1', '1')\n"
        )
        run = "print(is_equiv_minerva('2^{2^{325^{5}}}', '1'))"

        stdout = _run_with_memory_growth_ceiling(setup, run, ceiling_bytes=50 * 2**20, timeout_s=60)

        assert stdout.strip().splitlines()[-1] == "False"

    def test_sympy_does_not_compute_with_gmp(self) -> None:
        # The memory cap relies on an oversized allocation raising MemoryError. Once gmpy2 is installed, sympy
        # switches to GMP automatically, and GMP aborts the whole process instead: the power-tower check
        # would crash the eval again.
        assert GROUND_TYPES != "gmpy", "gmpy2 is installed; GMP aborts on the memory cap instead of raising"

    def test_metric_defaults_to_english_but_can_select_german(self) -> None:
        german = "Finale Antwort: Die finale Antwort lautet 24. Ich hoffe, die Antwort ist korrekt."
        default_results = MathMinervaCompletion().calculate(make_completion(german, "24"))

        german_metric = MathMinervaCompletion(cot_style="minerva_de")
        german_results = german_metric.calculate(make_completion(german, "24"))

        assert [result.value for result in default_results] == [0.0, 0.0]
        assert [result.value for result in german_results] == [1.0, 1.0]
