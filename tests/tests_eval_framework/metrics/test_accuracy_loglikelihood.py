import pytest

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyBayesianLoglikelihood,
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.shared.types import Error, Loglikelihood


def create_loglikelihood(
    loglikelihoods: dict[str, float],
    ground_truth: str | list[str],
    *,
    error: Error | None = None,
) -> Loglikelihood:
    return Loglikelihood(
        id=1,
        subject="test",
        ground_truth=ground_truth,
        prompt="test",
        prompt_num_tokens=None,
        loglikelihoods=loglikelihoods,
        loglikelihoods_num_tokens={},
        error=error,
    )


@pytest.mark.parametrize(
    "response,expected_value",
    [
        pytest.param(
            Loglikelihood(
                id=1,
                subject="test",
                ground_truth="A",
                prompt="test",
                prompt_num_tokens=None,
                loglikelihoods={"A": -0.1, "B": -0.5},
                loglikelihoods_num_tokens={"A": -1, "B": -1},
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
                prompt_num_tokens=None,
                loglikelihoods={" ": -1.0, "a": -2.0},
                loglikelihoods_num_tokens={" ": -1, "a": -1},
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


def test_accuracy_bayesian_estimates_centered_slope_and_corrects_scores() -> None:
    steep_response = create_loglikelihood({"a": -1.0, "aaa": -5.0}, "a")
    flat_response = create_loglikelihood({"a": -1.0, "aaa": -1.0}, ["aaa", "other"])
    metric = AccuracyBayesianLoglikelihood()

    metric.prepare([steep_response, flat_response])
    results = metric.calculate(flat_response)

    assert metric.length_decay == pytest.approx(-1.0)
    assert results[0].value == 1.0
    assert results[0].metric_name == "Accuracy Bayesian Loglikelihood"
    assert results[0].higher_is_better is True


def test_accuracy_bayesian_uses_utf8_byte_lengths() -> None:
    metric = AccuracyBayesianLoglikelihood()

    metric.prepare([create_loglikelihood({"é": -2.0, "aaa": -4.0}, "é")])

    assert metric.length_decay == pytest.approx(-2.0)


def test_accuracy_bayesian_ignores_errored_responses_when_estimating_slope() -> None:
    error = Error(error_class="RuntimeError", message="failed", traceback="trace")
    valid_response = create_loglikelihood({"a": -1.0, "aaa": -3.0}, "a")
    errored_response = create_loglikelihood({"a": -1.0, "aaa": -101.0}, "a", error=error)
    metric = AccuracyBayesianLoglikelihood()

    metric.prepare([valid_response, errored_response])

    assert metric.length_decay == pytest.approx(-1.0)


def test_accuracy_bayesian_falls_back_to_zero_without_length_variation() -> None:
    metric = AccuracyBayesianLoglikelihood()

    metric.prepare([create_loglikelihood({"a": -1.0, "b": -2.0}, "a")])

    assert metric.length_decay == 0.0
