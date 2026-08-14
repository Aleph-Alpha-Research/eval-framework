from eval_framework.metrics.efficiency.reasoning_tokens import NumReasoningTokens
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
    return Completion(**defaults)  # type: ignore[arg-type]


def test_num_reasoning_tokens_returns_reported_count() -> None:
    results = NumReasoningTokens().calculate(_completion())
    assert len(results) == 1
    assert results[0].metric_name == "NumReasoningTokens"
    assert results[0].value == 6
    assert results[0].higher_is_better is False


def test_num_reasoning_tokens_is_none_when_backend_did_not_report() -> None:
    results = NumReasoningTokens().calculate(_completion(raw_completion_reasoning_num_tokens=None))
    assert len(results) == 1
    assert results[0].value is None


def test_num_reasoning_tokens_is_none_when_sample_errored() -> None:
    error = Error(error_class="", message="", traceback="")
    results = NumReasoningTokens().calculate(_completion(error=error))
    assert len(results) == 1
    assert results[0].value is None
    assert results[0].error is error
