from eval_framework.metrics.efficiency.completion_tokens import NumCompletionTokens
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
        "raw_completion_num_tokens": 4,
    }
    defaults.update(overrides)
    return Completion(**defaults)  # type: ignore[arg-type]


def test_completion_tokens_returns_reported_count() -> None:
    results = NumCompletionTokens().calculate(_completion())
    assert len(results) == 1
    assert results[0].metric_name == "NumCompletionTokens"
    assert results[0].value == 4
    assert results[0].higher_is_better is False


def test_completion_tokens_is_none_when_backend_did_not_report() -> None:
    results = NumCompletionTokens().calculate(_completion(raw_completion_num_tokens=None))
    assert len(results) == 1
    assert results[0].value is None


def test_completion_tokens_is_none_when_sample_errored() -> None:
    error = Error(error_class="", message="", traceback="")
    results = NumCompletionTokens().calculate(_completion(error=error))
    assert len(results) == 1
    assert results[0].value is None
    assert results[0].error is error
