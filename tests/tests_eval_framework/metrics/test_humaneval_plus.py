from unittest.mock import patch

from eval_framework.metrics.completion.code_assertion import HumanEvalPlusCodeCompletionAssertion
from eval_framework.shared.types import Completion


def _completion(completion_text: str = "print(True)") -> Completion:
    return Completion(
        id=0,
        subject="no_subject",
        ground_truth=None,
        prompt_num_tokens=None,
        messages=None,
        prompt="",
        completion=completion_text,
        raw_completion="",
        raw_completion_num_tokens=None,
    )


def test_calculate_installs_numpy_into_the_sandbox() -> None:
    # Given a completion, so run_python_code gets called; calculate() itself is inherited and
    # already covered by test_code_assertion.py
    metric = HumanEvalPlusCodeCompletionAssertion()

    # When
    with patch("eval_framework.metrics.completion.code_assertion.run_python_code", return_value="True") as mock_run:
        metric.calculate(_completion())

    # Then numpy is requested for the sandbox, unlike the base CodeCompletionAssertion
    assert mock_run.call_args.kwargs["packages"] == ["numpy"]
