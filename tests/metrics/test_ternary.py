import pytest

from eval_framework.metrics.loglikelihood.ternary import TernaryScore
from eval_framework.shared.types import Loglikelihood


def make_response(loglikelihoods, ground_truth, error=None):
    return Loglikelihood(
        id=1,
        subject="test",
        ground_truth=ground_truth,
        prompt="test",
        prompt_sequence_positions=None,
        loglikelihoods=loglikelihoods,
        loglikelihoods_sequence_positions={k: -1 for k in loglikelihoods},
        error=error,
    )


@pytest.mark.parametrize(
    "loglikelihoods,ground_truth,expected",
    [
        ({"A": 0.0, "B": -1.0, "IDK": -2.0}, "A", 1.0),
        ({"A": -1.0, "B": 0.0, "IDK": -2.0}, "A", -1.0),
        ({"A": -2.0, "B": -2.0, "IDK": 0.0}, "A", 0.0),
    ],
)
def test_ternary_score_cases(loglikelihoods, ground_truth, expected):
    metric = TernaryScore()
    response = make_response(loglikelihoods, ground_truth)
    result = metric.calculate(response)[0]
    assert result.value == expected


def test_ternary_score_normalisation():
    metric = TernaryScore()
    response = make_response({" a ": 0.0, "B": -1.0, "IDK": -2.0}, "A")
    result = metric.calculate(response)[0]
    assert result.value == 1.0


def test_ternary_score_error():
    from eval_framework.shared.types import Error

    metric = TernaryScore()
    err = Error(error_class="fail", message="fail", traceback="")
    response = make_response({"A": -1.0}, "A", error=err)
    result = metric.calculate(response)[0]
    assert result.value is None
    assert result.error == err
