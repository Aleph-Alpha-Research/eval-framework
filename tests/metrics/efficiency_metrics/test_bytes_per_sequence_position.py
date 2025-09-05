import pytest

from eval_framework.metrics.efficiency.bytes_per_sequence_position import (
    BytesCompletion,
    BytesLoglikelihood,
    SequencePositionsCompletion,
    SequencePositionsLoglikelihood,
)
from eval_framework.shared.types import Completion, Error, Loglikelihood

log_likelihood_params = [
    pytest.param(
        Loglikelihood(
            id=1,
            subject="x",
            ground_truth="big car",
            prompt="test",
            prompt_sequence_positions=1,
            loglikelihoods={"big car": -1.0, "small mouse house": -2.0},
            loglikelihoods_sequence_positions={"big car": 2, "small mouse house": 3},
        ),
        1 + 2 + 3,
        4 + 7 + 17,
        id="all_filled",
    ),
    pytest.param(
        Loglikelihood(
            id=1,
            subject="x",
            ground_truth="big car",
            prompt="test",
            prompt_sequence_positions=1,
            loglikelihoods={"big car": -1.0, "small mouse house": -2.0},
            loglikelihoods_sequence_positions={},
        ),
        None,
        None,
        id="loglikelihoods_None",
    ),
    pytest.param(
        Loglikelihood(
            id=1,
            subject="x",
            ground_truth="big car",
            prompt="test",
            prompt_sequence_positions=None,
            loglikelihoods={},
            loglikelihoods_sequence_positions={"big car": 2, "small mouse house": 3},
        ),
        None,
        None,
        id="prompt_None",
    ),
    pytest.param(
        Loglikelihood(
            id=1,
            subject="x",
            ground_truth="big car",
            prompt="test",
            prompt_sequence_positions=1,
            loglikelihoods={"big car": -1.0, "small mouse house": -2.0},
            loglikelihoods_sequence_positions={"big car": 2, "small mouse house": 3},
            error=Error(error_class="", message="", traceback=""),
        ),
        None,
        None,
        id="with_error",
    ),
]


@pytest.mark.parametrize("response,expected_sp,expected_bytes", log_likelihood_params)
def test_sequence_positions_loglikelihood(
    response: Loglikelihood, expected_sp: float | None, expected_bytes: float | None
) -> None:
    metric = SequencePositionsLoglikelihood()
    results = metric.calculate(response)
    assert len(results) == 1
    assert results[0].value == expected_sp


@pytest.mark.parametrize("response,expected_sp,expected_bytes", log_likelihood_params)
def test_bytes_loglikelihood(response: Loglikelihood, expected_sp: float | None, expected_bytes: float | None) -> None:
    metric = BytesLoglikelihood()
    results = metric.calculate(response)
    assert len(results) == 1
    assert results[0].value == expected_bytes


completion_params = [
    pytest.param(
        Completion(
            id=1,
            subject="x",
            ground_truth="wow, a cute dog!",
            prompt="test",
            prompt_sequence_positions=1,
            messages=None,
            completion="25",
            raw_completion="the answer is 25",
            raw_completion_sequence_positions=4,
        ),
        1 + 4,
        4 + 16,
        id="all_filled",
    ),
    pytest.param(
        Completion(
            id=1,
            subject="x",
            ground_truth="wow, a cute dog!",
            prompt="test",
            prompt_sequence_positions=1,
            messages=None,
            completion="25",
            raw_completion="the answer is 25",
            raw_completion_sequence_positions=None,
        ),
        None,
        None,
        id="completion_None",
    ),
    pytest.param(
        Completion(
            id=1,
            subject="x",
            ground_truth="wow, a cute dog!",
            prompt="test",
            prompt_sequence_positions=None,
            messages=None,
            completion="25",
            raw_completion="the answer is 25",
            raw_completion_sequence_positions=4,
        ),
        None,
        None,
        id="prompt_None",
    ),
    pytest.param(
        Completion(
            id=1,
            subject="x",
            ground_truth="wow, a cute dog!",
            prompt="test",
            prompt_sequence_positions=1,
            messages=None,
            completion="25",
            raw_completion="the answer is 25",
            raw_completion_sequence_positions=4,
            error=Error(error_class="", message="", traceback=""),
        ),
        None,
        None,
        id="with_error",
    ),
]


@pytest.mark.parametrize("response,expected_sp,expected_bytes", completion_params)
def test_sequence_positions_completion(
    response: Completion, expected_sp: float | None, expected_bytes: float | None
) -> None:
    metric = SequencePositionsCompletion()
    results = metric.calculate(response)
    assert len(results) == 1
    assert results[0].value == expected_sp


@pytest.mark.parametrize("response,expected_sp,expected_bytes", completion_params)
def test_bytes_completion(response: Completion, expected_sp: float | None, expected_bytes: float | None) -> None:
    metric = BytesCompletion()
    results = metric.calculate(response)
    assert len(results) == 1
    assert results[0].value == expected_bytes
