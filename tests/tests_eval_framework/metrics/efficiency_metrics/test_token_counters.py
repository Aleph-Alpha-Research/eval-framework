from eval_framework.metrics.efficiency.token_counters import TokenCounts
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
        "raw_completion_reasoning_num_tokens": 6,
    }
    defaults.update(overrides)
    return Completion(**defaults)


def test_token_counts_returns_reported_counts() -> None:
    results = TokenCounts().calculate(_completion())
    assert len(results) == 2

    by_name = {result.metric_name: result for result in results}
    assert by_name["TokenCounts/Completion"].value == 10
    assert by_name["TokenCounts/Completion"].higher_is_better is False
    assert by_name["TokenCounts/Reasoning"].value == 6
    assert by_name["TokenCounts/Reasoning"].higher_is_better is False


def test_token_counts_completion_is_none_when_backend_did_not_report() -> None:
    results = TokenCounts().calculate(_completion(raw_completion_num_tokens=None))
    by_name = {result.metric_name: result for result in results}
    assert by_name["TokenCounts/Completion"].value is None
    assert by_name["TokenCounts/Reasoning"].value == 6


def test_token_counts_reasoning_is_none_when_backend_did_not_report() -> None:
    results = TokenCounts().calculate(_completion(raw_completion_reasoning_num_tokens=None))
    by_name = {result.metric_name: result for result in results}
    assert by_name["TokenCounts/Completion"].value == 10
    assert by_name["TokenCounts/Reasoning"].value is None


def test_token_counts_are_none_when_sample_errored() -> None:
    error = Error(error_class="", message="", traceback="")
    results = TokenCounts().calculate(_completion(error=error))
    by_name = {result.metric_name: result for result in results}
    assert by_name["TokenCounts/Completion"].value is None
    assert by_name["TokenCounts/Reasoning"].value is None
    assert all(result.error is error for result in results)
