import pytest

from eval_framework.metrics.completion_metrics.concordance_index import ConcordanceIndex
from eval_framework.shared.types import Completion


@pytest.mark.parametrize(
    "response,expected_value",
    [
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="[0, 1, 2, 3]",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="[0, 1, 2, 3]",
                raw_completion="[0, 1, 2, 3]",
                raw_completion_sequence_positions=None,
            ),
            1.0,
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="[0, 1, 2, 3]",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="[0, 2, 1, 3]",
                raw_completion="[0, 2, 1, 3]",
                raw_completion_sequence_positions=None,
            ),
            0.5,
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="[0, 1, 2, 3]",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="[3, 2, 1, 0]",
                raw_completion="[3, 2, 1, 0]",
                raw_completion_sequence_positions=None,
            ),
            0.0,
        ),
    ],
)
def test_concordance_index(response: Completion, expected_value: float) -> None:
    assert ConcordanceIndex().calculate(response)[0].value == expected_value
