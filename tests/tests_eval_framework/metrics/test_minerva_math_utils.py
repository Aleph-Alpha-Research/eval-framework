import resource

import pytest

from eval_framework.metrics.completion.minerva_math_utils import is_equiv_minerva


class TestIsEquivMinerva:
    def test_equivalent_integers(self) -> None:
        assert is_equiv_minerva("1", "1") is True

    def test_non_equivalent_integers(self) -> None:
        assert is_equiv_minerva("1", "2") is False

    def test_equivalent_polynomials(self) -> None:
        assert is_equiv_minerva("x^2", "x^2") is True

    def test_invalid_latex_returns_false(self) -> None:
        assert is_equiv_minerva("???", "!!!") is False

    def test_timeout_returns_false(self) -> None:
        assert is_equiv_minerva("x", "y", timeout_seconds=0) is False

    @pytest.mark.cpu_slow
    def test_pathological_expression_does_not_oom_parent(self) -> None:
        """A product-of-binomials fraction forces SymPy's simplify() to expand
        a polynomial with 2^n terms.  At n=15 the intermediate representation
        exceeds 3 GiB (measured empirically) — well above the 2 GiB subprocess
        memory budget.  Without the subprocess memory limit, this OOM-kills the
        evaluation process.

        The test verifies that:
        1. is_equiv_minerva returns False (not crash/hang),
        2. the parent process's memory does not grow (the blowup stays in the
           subprocess).
        """
        # Given
        n = 15
        num = "".join(f"(a_{{{i}}} + b_{{{i}}})" for i in range(n))
        den = "".join(f"(c_{{{i}}} + d_{{{i}}})" for i in range(n))
        pathological = rf"\frac{{{num}}}{{{den}}}"

        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        # When — use a long timeout so that simplify() has time to reach
        # the expensive cancel() path which triggers polynomial expansion.
        result = is_equiv_minerva(pathological, "1", timeout_seconds=30)

        # Then
        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_growth_mb = (rss_after - rss_before) / (1024 * 1024)

        assert result is False
        assert rss_growth_mb < 100, (
            f"Parent process RSS grew by {rss_growth_mb:.0f} MB; "
            f"memory from the pathological expression leaked out of the subprocess"
        )
