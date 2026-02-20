"""Tests comparing our _linear_sum_assignment (Hungarian) implementation to scipy's.

Scipy is a dev-only dependency; these tests are skipped when scipy is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval_framework.external.drop_process_results import _linear_sum_assignment

scipy_linear_sum_assignment = pytest.importorskip("scipy").optimize.linear_sum_assignment


def _cost_of_assignment(cost_matrix: np.ndarray, row_ind: np.ndarray, col_ind: np.ndarray) -> float:
    return float(cost_matrix[row_ind, col_ind].sum())


@pytest.mark.parametrize(
    "cost_matrix",
    [
        # Scipy doc example: optimal cost = 5
        np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]]),
        # Square, simple
        np.array([[1, 2], [3, 4]]),
        np.array([[10, 0], [0, 10]]),
        # Rectangular: more rows than columns
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
        np.array([[1, 2], [3, 4], [5, 6]]),
        # Rectangular: more columns than rows
        np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
        np.array([[0, 1], [1, 0], [2, 2]]),
        # Single row/column
        np.array([[1, 2, 3]]),
        np.array([[1], [2], [3]]),
        # With zeros (degenerate)
        np.array([[0, 0], [0, 0]]),
        np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]]),
        # DROP-style: scores as costs (we pass -scores to maximize)
        np.array([[0.8, 0.1], [0.2, 0.9]]),
    ],
)
def test_linear_sum_assignment_matches_scipy(cost_matrix: np.ndarray) -> None:
    """For the same cost matrix, our implementation and scipy's give the same optimal cost."""
    cost = np.asarray(cost_matrix, dtype=np.float64)

    our_row, our_col = _linear_sum_assignment(cost)
    scipy_row, scipy_col = scipy_linear_sum_assignment(cost)

    our_cost = _cost_of_assignment(cost, our_row, our_col)
    scipy_cost = _cost_of_assignment(cost, scipy_row, scipy_col)

    assert our_cost == scipy_cost, (
        f"Optimal cost mismatch: ours={our_cost} (row={our_row}, col={our_col}), "
        f"scipy={scipy_cost} (row={scipy_row}, col={scipy_col})"
    )
    # Both should have row indices sorted (our API; scipy's too for square)
    assert np.all(our_row == np.sort(our_row)), "Our row_ind should be sorted"
