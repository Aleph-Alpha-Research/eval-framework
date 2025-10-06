import pytest
from eval_framework.metrics.loglikelihood.confidence_weighted_accuracy import ConfidenceWeightedAccuracy
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
        ({"A": -0.1, "B": -0.5}, "A", 0.599),
        ({"A": -0.1, "B": -0.5}, "B", 0.0),
        ({"A": 0.0, "B": -1.0, "C": -2.0}, "A", 0.665),
    ],
)
def test_confidence_weighted_accuracy_cases(loglikelihoods, ground_truth, expected):
    metric = ConfidenceWeightedAccuracy()
    response = make_response(loglikelihoods, ground_truth)
    result = metric.calculate(response)[0]
    assert result.value == pytest.approx(expected, rel=1e-3)
    assert result.metric_name == "Confidence-weighted Accuracy"
    assert result.higher_is_better is True

def test_confidence_weighted_normalise():
    metric = ConfidenceWeightedAccuracy()
    loglikelihoods = {" a ": 0.0, "B": -1.0, "c ": -2.0}
    response = make_response(loglikelihoods, "A")
    result = metric.calculate(response)[0]
    assert result.value == pytest.approx(0.576, rel=1e-3)

def test_confidence_weighted_accuracy_error():
    from eval_framework.shared.types import Error
    metric = ConfidenceWeightedAccuracy()
    err = Error(error_class="fail", message="fail", traceback="")
    response = make_response({"A": -1.0}, "A", error=err)
    result = metric.calculate(response)[0]
    assert result.value is None
    assert result.error == err
