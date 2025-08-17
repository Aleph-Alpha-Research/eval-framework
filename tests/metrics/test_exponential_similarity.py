import pytest

from eval_framework.metrics.completion_metrics.exponential_similarity import ExponentialSimilarity
from eval_framework.shared.types import Completion


@pytest.mark.parametrize(
    "response,expected_value",
    [
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="80",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="80",
                raw_completion="80",
                raw_completion_sequence_positions=None,
            ),
            1.0,
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="75",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="65",
                raw_completion="65",
                raw_completion_sequence_positions=None,
            ),
            0.5000000000000001,
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="75",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="70",
                raw_completion="70",
                raw_completion_sequence_positions=None,
            ),
            0.7071067811865474,
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="not a number",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="also not a number",
                raw_completion="also not a number",
                raw_completion_sequence_positions=None,
            ),
            None,  # Expecting None as the value due to conversion error
            id="non_numeric_values",
        ),
    ],
)
def test_exponential_similarity(response: Completion, expected_value: float) -> None:
    result = ExponentialSimilarity().calculate(response)[0]
    assert result.value == expected_value
    if expected_value is None:
        assert result.error is not None
    else:
        assert result.error is None
