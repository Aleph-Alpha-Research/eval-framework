import pytest

from eval_framework.metrics.completion_metrics.placeholder_checker import (
    PlaceholderChecker,
    PlaceholderCheckerMetricContext,
)
from eval_framework.shared.types import Completion


@pytest.mark.parametrize(
    "response,expected_value",
    [
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth=None,
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="[person] and [person] live at [address].",
                raw_completion="[person] and [person] live at [address].",
                raw_completion_sequence_positions=None,
                context=PlaceholderCheckerMetricContext(num_placeholders=3),
            ),
            1.0,
            id="placeholder_checker_exact_number",
        ),
        pytest.param(
            Completion(
                id=29,
                subject="test",
                ground_truth=None,
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="[person] and [person] live at [address].",
                raw_completion="[person] and [person] live at [address].",
                raw_completion_sequence_positions=None,
                context=PlaceholderCheckerMetricContext(num_placeholders=4),
            ),
            0.0,
            id="placeholder_checker_not_enough_placeholders",
        ),
        pytest.param(
            Completion(
                id=28,
                subject="test",
                ground_truth=None,
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="[person] and [person] live at [address].",
                raw_completion="[person] and [person] live at [address].",
                raw_completion_sequence_positions=None,
                context=PlaceholderCheckerMetricContext(num_placeholders=2),
            ),
            1.0,
            id="placeholder_checker_more_placeholders",
        ),
        pytest.param(
            Completion(
                id=27,
                subject="test",
                ground_truth=None,
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="No placeholders here.",
                raw_completion="No placeholders here.",
                raw_completion_sequence_positions=None,
                context=PlaceholderCheckerMetricContext(num_placeholders=1),
            ),
            0.0,
            id="placeholder_checker_no_placeholders",
        ),
    ],
)
def test_placeholder_checker(response: Completion, expected_value: float) -> None:
    metric = PlaceholderChecker()
    results = metric.calculate(response)
    assert len(results) == 1
    assert results[0].value == pytest.approx(expected_value)
    assert results[0].metric_name == "Placeholder Check"
    assert results[0].higher_is_better is True
