import pytest

from eval_framework.exceptions import LogicError
from eval_framework.metrics.completion.ter import TER
from eval_framework.shared.types import Completion


@pytest.mark.parametrize(
    "response,expected_value",
    [
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="wow, a cute dog!",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="wow, an ugly cat!",
                raw_completion="wow, an ugly cat!",
                raw_completion_sequence_positions=None,
            ),
            75.0,
            id="ter_somewhat_different",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="wow, a cute dog!",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="wow, a cute dog!",
                raw_completion="wow, a cute dog!",
                raw_completion_sequence_positions=None,
            ),
            0.0,
            id="ter_exact_match",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="holy moly, what kind of a rocket is that?",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="oh this is a cute dog!",
                raw_completion="oh this is a cute dog!",
                raw_completion_sequence_positions=None,
            ),
            88.88888888888889,
            id="ter_completely_different",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="some ground truth",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="",
                raw_completion="",
                raw_completion_sequence_positions=None,
            ),
            100,
            id="ter_mising_completion_is_maximally_different",
        ),
    ],
)
def test_ter(response: Completion, expected_value: float) -> None:
    metric = TER()
    results = metric.calculate(response)
    assert len(results) == 1
    assert results[0].value == pytest.approx(expected_value)
    assert results[0].metric_name == "TER"
    assert results[0].higher_is_better is False


@pytest.mark.parametrize(
    "completion",
    [
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="A",
                raw_completion="A",
                raw_completion_sequence_positions=None,
            ),
            id="ter_missing_ground_truth",
        ),
    ],
)
def test_ter_errors(completion: Completion) -> None:
    metric = TER()
    with pytest.raises(LogicError):
        metric.calculate(completion)
