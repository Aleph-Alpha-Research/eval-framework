import pytest

from eval_framework.metrics.completion_metrics.f1 import F1, calculate_f1
from eval_framework.shared.types import Completion


@pytest.mark.parametrize(
    "reference, hypothesis, expected",
    [
        pytest.param("the cat sat".lower().split(), "the cat sat".lower().split(), 1.0, id="perfect_match"),
        pytest.param("the cat sat".lower().split(), "dog runs fast".lower().split(), 0.0, id="no_match"),
        pytest.param(
            "the cat sat".lower().split(),
            "the cat runs".lower().split(),
            2 / 3,
            id="partial_match",
        ),
        pytest.param("The Cat Sat".lower().split(), "the cat sat".lower().split(), 1.0, id="case_insensitive"),
        pytest.param("", "", 1.0, id="empty_strings"),
        pytest.param("the cat".lower().split(), "", 0.0, id="empty_hypothesis"),
        pytest.param("the  cat  sat".lower().split(), "the cat sat".lower().split(), 1.0, id="extra_whitespace"),
    ],
)
def test_calculate_f1_function(reference: list[str], hypothesis: list[str], expected: float) -> None:
    assert calculate_f1(reference, hypothesis) == expected


@pytest.mark.parametrize(
    "ground_truth, completion, expected",
    [
        pytest.param("the cat sat", "the cat sat", 1.0, id="single_ground_truth_perfect"),
        pytest.param(
            ["the cat sat", "a cat sat", "the cat rested"], "the cat sat", 1.0, id="multiple_ground_truths_perfect"
        ),
        pytest.param(
            ["the cat sat", "a dog ran"],
            "the cat ran",
            2 / 3,
            id="multiple_ground_truths_partial",
        ),
        pytest.param(None, "the cat sat", 0.0, id="none_ground_truth"),
        pytest.param([], "the cat sat", 0.0, id="empty_ground_truth_list"),
        pytest.param(["The Cat Sat", "A DOG RAN"], "the cat sat", 1.0, id="case_insensitive"),
    ],
)
def test_f1_metric_class(ground_truth: str | list[str] | None, completion: str, expected: float) -> None:
    metric = F1()
    completion_obj = Completion(
        id=1,
        subject="test",
        ground_truth=ground_truth,
        prompt="test prompt",
        prompt_sequence_positions=None,
        messages=None,
        completion=completion,
        raw_completion=completion,
        raw_completion_sequence_positions=None,
    )

    results = metric.calculate(completion_obj)
    assert len(results) == 1
    assert pytest.approx(results[0].value) == expected
    assert results[0].metric_name == "F1"
    assert results[0].higher_is_better is True
