import pytest

from eval_framework.metrics.completion_metrics.accuracy_completion import AccuracyCompletion
from eval_framework.shared.types import Completion


@pytest.mark.parametrize(
    "response,expected_value",
    [
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="correct",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="correct",
                raw_completion="correct",
                raw_completion_sequence_positions=None,
            ),
            1.0,
            id="acc_exact_match",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="correct",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="wrong",
                raw_completion="wrong",
                raw_completion_sequence_positions=None,
            ),
            0.0,
            id="acc_no_match",
        ),
    ],
)
def test_accuracy_completion(response: Completion, expected_value: float) -> None:
    metric = AccuracyCompletion()
    results = metric.calculate(response)
    assert len(results) == 1
    assert results[0].value == pytest.approx(expected_value)
    assert results[0].metric_name == "Accuracy Completion"
    assert results[0].higher_is_better is True
