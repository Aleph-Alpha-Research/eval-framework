import pytest

from eval_framework.tasks.benchmarks.zero_scrolls import (
    ZERO_SCROLLS_SPACE_DIGEST,
)


@pytest.mark.parametrize(
    "completion_text,expected_result",
    [
        # Basic percentage formats
        ("30%", "30"),
        ("30.5%", "30.5"),
        # Just numbers
        ("60", "60"),
        ("42.5", "42.5"),
        ("  75  ", "75"),
        # Simple phrases
        ("it's 60", "60"),
        ("its 45.5", "45.5"),
        ("that's 33", "33"),
        ("thats 22.5", "22.5"),
        # Percentage word formats
        ("30 percent", "30"),
        ("30.5 percent", "30.5"),
        ("The percentage is 42", "42"),
        ("The percentage: 75.5", "75.5"),
        # More complex sentences
        ("Based on the data, the percentage is approximately 65%", "65"),
        ("I would estimate that it equals 28.3 percent", "28.3"),
        ("After analyzing the information, it is about 50 percent", "50"),
        ("The correct answer is roughly 33.3%", "33.3"),
        # Fallback to finding any number
        ("The value we're looking for is 42", "42"),
        ("According to calculations, 37.5 is the answer", "37.5"),
        # Edge cases
        ("No numbers here", "No numbers here"),
        ("", ""),
    ],
)
def test_post_process_generated_completion(completion_text: str, expected_result: str) -> None:
    task = ZERO_SCROLLS_SPACE_DIGEST()
    result = task.post_process_generated_completion(completion_text)
    assert result == expected_result
