import pytest

from eval_framework.metrics.completion.repetition import WordRepetition
from eval_framework.shared.types import Completion


@pytest.mark.parametrize(
    "completion_str, window_size, min_repetitions, expected_value",
    [
        ("foo bar baz", 1, 1, 0.0),  # No repetitions
        ("Word " * 3, 1, 1, 1.0),  # Single word repeated
        ("Word " * 3, 1, 4, 0.0),  # Single word repetition, but not enough repetitions
        ("Word " * 3, 2, 1, 1.0),  # Multiple words repeated
        ("Word " * 3, 2, 2, 0.0),  # Multiple words, but not enough repetitions
        ("Word " * 3, 4, 1, 0.0),  # Larger window size than word count
        ("foo|foo", 1, 1, 1.0),  # Words separated by punctuation
        ("", 1, 1, 0.0),  # Empty input
        ("???", 1, 1, 0.0),  # Empty input with punctuation
    ],
)
def test_word_repetition(
    completion_str: str,
    window_size: int,
    min_repetitions: int,
    expected_value: float,
) -> None:
    # Given
    completion = Completion(
        id=0,
        subject="en",
        ground_truth=None,
        prompt="",
        prompt_sequence_positions=None,
        messages=None,
        completion=completion_str,
        raw_completion=completion_str,
        raw_completion_sequence_positions=None,
    )
    metric = WordRepetition(window_size=window_size, min_repetitions=min_repetitions)

    # When
    results = metric.calculate(completion)

    # Then
    assert len(results) == 1
    assert results[0].value == expected_value
