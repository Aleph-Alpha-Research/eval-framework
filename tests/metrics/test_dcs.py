import pytest
from eval_framework.metrics.loglikelihood.dcs import DistributionalCorrectnessScore
from eval_framework.shared.types import Error, Loglikelihood


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
        ({"A": 0.0, "IDK": -100.0}, "A", 1.0),
        ({"A": -100.0, "B": 0.0, "IDK": -100.0}, "A", -1.0),
        ({"A": -100.0, "B": -100.0, "IDK": 0.0}, "A", 0.0),
        ({"A": 0.0, "B": 0.0, "IDK": -100.0}, "A", 0.0),
    ],
)
def test_dcs_loglikelihood_cases(loglikelihoods, ground_truth, expected):
    metric = DistributionalCorrectnessScore()
    response = make_response(loglikelihoods, ground_truth)
    result = metric.calculate(response)[0]
    assert result.value == pytest.approx(expected)

@pytest.mark.parametrize(
    "probs,ground_truth,expected",
    [
        ({"A": 0.31, "B": 0.27, "C": 0.40, "D": 0.01, "IDK": 0.01}, "C", -0.19),
        ({"A": 0.15, "B": 0.05, "C": 0.40, "D": 0.01, "IDK": 0.39}, "C", 0.19),
        ({"A": 0.25, "B": 0.24, "C": 0.26, "D": 0.24, "IDK": 0.01}, "C", -0.47),
        ({"A": 0.01, "B": 0.01, "C": 0.96, "D": 0.01, "IDK": 0.01}, "C", 0.93),
    ],
)
def test_dcs_probs_cases(probs, ground_truth, expected):
    metric = DistributionalCorrectnessScore()
    response = make_response({k: float('nan') for k in probs}, ground_truth)
    def fake_softmax(_): return probs
    metric._softmax = fake_softmax
    result = metric.calculate(response)[0]
    assert result.value == pytest.approx(expected, rel=1e-6)

def test_dcs_normalisation():
    metric = DistributionalCorrectnessScore()
    probs = {"a ": 0.01, " B": 0.01, "C": 0.96, " d ": 0.01, "IDK": 0.01}
    response = make_response({k: float('nan') for k in probs}, "C")
    def fake_softmax(_): return probs
    metric._softmax = fake_softmax
    result = metric.calculate(response)[0]
    assert result.value == pytest.approx(0.93, rel=1e-6)

def test_dcs_error():
    from eval_framework.shared.types import Error
    metric = DistributionalCorrectnessScore()
    err = Error(error_class="fail", message="fail", traceback="")
    response = make_response({"A": -1.0}, "A", error=err)
    result = metric.calculate(response)[0]
    assert result.value is None
    assert result.error == err