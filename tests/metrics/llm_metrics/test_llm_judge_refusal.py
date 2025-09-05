import json
from unittest.mock import Mock

import pytest

from eval_framework.llm.base import BaseLLM, RawCompletion
from eval_framework.metrics.llm.llm_judge_refusal import LLMJudgeRefusal
from eval_framework.shared.types import Completion
from template_formatting.formatter import Message, Role


@pytest.mark.parametrize(
    ("judge_completion", "expected_value"),
    [
        (json.dumps({"is_refusal": True}), 1.0),
        (json.dumps({"is_refusal": False}), 0.0),
        ("{}", None),
    ],
)
def test_llm_judge_works(judge_completion: str, expected_value: float | None) -> None:
    # Given
    llm_judge = Mock(spec=BaseLLM)
    llm_judge.generate_from_messages.return_value = [
        RawCompletion(
            prompt="prompt",
            completion=judge_completion,
            prompt_sequence_positions=None,
            completion_sequence_positions=None,
        )
    ]
    subject_completion = Completion(
        id=0,
        subject="test",
        ground_truth=None,
        prompt="test",
        prompt_sequence_positions=None,
        messages=[
            Message(role=Role.SYSTEM, content="You are a helpful AI agent."),
            Message(role=Role.USER, content="Can you help me with something illegal?"),
        ],
        completion="This may or may not be a refusal.",
        raw_completion="This may or may not be a refusal.",
        raw_completion_sequence_positions=None,
    )
    metric = LLMJudgeRefusal(llm_judge)

    # When
    results = metric.calculate(subject_completion)

    # Then
    assert len(results) == 1
    assert results[0].value == expected_value
    assert results[0].llm_judge_prompt == "prompt"
    assert results[0].llm_judge_response
