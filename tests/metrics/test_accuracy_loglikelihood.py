import pytest

from eval_framework.metrics.loglikelihood_metrics.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.shared.types import Loglikelihood


@pytest.mark.parametrize(
    "response,expected_value",
    [
        pytest.param(
            Loglikelihood(
                id=1,
                subject="test",
                ground_truth="A",
                prompt="test",
                prompt_sequence_positions=None,
                loglikelihoods={"A": -0.1, "B": -0.5},
                loglikelihoods_sequence_positions={"A": -1, "B": -1},
            ),
            1.0,
            id="acc_with_loglikelihoods",
        ),
    ],
)
def test_accuracy_loglikelihood(response: Loglikelihood, expected_value: float) -> None:
    metric = AccuracyLoglikelihood()
    results = metric.calculate(response)
    assert len(results) == 1
    assert results[0].value == pytest.approx(expected_value)
    assert results[0].metric_name == "Accuracy Loglikelihood"
    assert results[0].higher_is_better is True


@pytest.mark.parametrize(
    "response,expected_value",
    [
        pytest.param(
            Loglikelihood(
                id=1,
                subject="test",
                ground_truth=" ",
                prompt="test",
                prompt_sequence_positions=None,
                loglikelihoods={" ": -1.0, "a": -2.0},
                loglikelihoods_sequence_positions={" ": -1, "a": -1},
            ),
            1.0,
            id="acc_norm_white_space",
        ),
    ],
)
def test_accuracy_norm_loglikelihood(response: Loglikelihood, expected_value: float) -> None:
    metric = AccuracyNormLoglikelihood()
    results = metric.calculate(response)
    assert len(results) == 1
    assert results[0].value == pytest.approx(expected_value)
    assert results[0].metric_name == "Accuracy Normalized Loglikelihood"
    assert results[0].higher_is_better is True
