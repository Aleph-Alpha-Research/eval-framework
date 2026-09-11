import pytest

from eval_framework.metrics.efficiency.finish_reason import FinishReason
from eval_framework.shared.types import Completion, Error


def _completion(**overrides: object) -> Completion:
    defaults: dict = {
        "id": 1,
        "subject": "x",
        "ground_truth": "gt",
        "prompt": "test",
        "prompt_num_tokens": 1,
        "messages": None,
        "completion": "42",
        "raw_completion": "the answer is 42",
        "raw_completion_num_tokens": 10,
        "raw_completion_finish_reason": "stop",
    }
    defaults.update(overrides)
    return Completion(**defaults)


@pytest.mark.parametrize(
    "finish_reason,expected",
    [
        ("stop", {"Stop": 1.0, "Length": 0.0, "Repetition": 0.0}),
        ("length", {"Stop": 0.0, "Length": 1.0, "Repetition": 0.0}),
        ("repetition", {"Stop": 0.0, "Length": 0.0, "Repetition": 1.0}),
        ("REPETITION", {"Stop": 0.0, "Length": 0.0, "Repetition": 1.0}),
        ("tool_calls", {"Stop": 0.0, "Length": 0.0, "Repetition": 0.0}),
    ],
)
def test_finish_reason_one_hot_encodes_reported_reason(finish_reason: str, expected: dict[str, float]) -> None:
    results = FinishReason().calculate(_completion(raw_completion_finish_reason=finish_reason))
    assert len(results) == 3

    by_name = {result.metric_name: result for result in results}
    assert {name.removeprefix("FinishReason/"): result.value for name, result in by_name.items()} == expected
    assert by_name["FinishReason/Stop"].higher_is_better is True
    assert by_name["FinishReason/Length"].higher_is_better is False
    assert by_name["FinishReason/Repetition"].higher_is_better is False


def test_finish_reason_is_none_when_backend_did_not_report() -> None:
    results = FinishReason().calculate(_completion(raw_completion_finish_reason=None))
    assert len(results) == 3
    assert all(result.value is None for result in results)
    assert all(result.error is None for result in results)


def test_finish_reason_is_none_when_sample_errored() -> None:
    error = Error(error_class="", message="", traceback="")
    results = FinishReason().calculate(_completion(raw_completion_finish_reason="repetition", error=error))
    assert len(results) == 3
    assert all(result.value is None for result in results)
    assert all(result.error is error for result in results)
