import numpy as np
import pytest

from eval_framework.metrics.loglikelihood.probability_mass import ProbabilityMass, ProbabilityMassNorm
from eval_framework.shared.types import Loglikelihood


@pytest.mark.parametrize(
    "response,expected_value",
    [
        pytest.param(
            Loglikelihood(
                id=1,
                subject="test",
                ground_truth=["A"],
                prompt="test",
                prompt_sequence_positions=None,
                loglikelihoods={"A": np.log(0.5), "B": np.log(0.3), "C": np.log(0.2)},
                loglikelihoods_sequence_positions={"A": -1, "B": -1, "C": -1},
            ),
            0.5,
            id="prob_mass_simple",
        ),
        pytest.param(
            Loglikelihood(
                id=1,
                subject="test",
                ground_truth=["C"],
                prompt="test",
                prompt_sequence_positions=None,
                loglikelihoods={"A": np.log(0.2), "B": np.log(0.3), "C": np.log(0.5)},
                loglikelihoods_sequence_positions={"A": -1, "B": -1, "C": -1},
            ),
            0.5,
            id="prob_mass_last_position",
        ),
    ],
)
def test_probability_mass(response: Loglikelihood, expected_value: float) -> None:
    metric = ProbabilityMass()
    results = metric.calculate(response)
    assert len(results) == 1
    assert results[0].value == pytest.approx(expected_value)
    assert results[0].metric_name == "Probability Mass"
    assert results[0].higher_is_better is True


@pytest.mark.parametrize(
    "response,expected_value",
    [
        pytest.param(
            Loglikelihood(
                id=1,
                subject="test",
                ground_truth=["long_answer"],
                prompt="test",
                prompt_sequence_positions=None,
                loglikelihoods={"long_answer": -4.0, "short": -1.0},
                loglikelihoods_sequence_positions={"long_answer": -1, "short": -1},
            ),
            0.45918195006784246,
            id="prob_mass_norm_length_normalization",
        ),
        pytest.param(
            Loglikelihood(
                id=1,
                subject="test",
                ground_truth=[""],
                prompt="test",
                prompt_sequence_positions=None,
                loglikelihoods={"": -1.0, "text": -4.0},
                loglikelihoods_sequence_positions={"": -1, "text": -1},
            ),
            0.5,
            id="prob_mass_norm_empty_string",
        ),
    ],
)
def test_probability_mass_norm(response: Loglikelihood, expected_value: float) -> None:
    metric = ProbabilityMassNorm()
    results = metric.calculate(response)
    assert len(results) == 1
    assert results[0].value == pytest.approx(expected_value)
    assert results[0].metric_name == "Probability Mass Normalized"
    assert results[0].higher_is_better is True
