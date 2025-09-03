import json
from unittest.mock import Mock

from eval_framework.llm.base import BaseLLM
from eval_framework.metrics.llm.llm_judge_completion_accuracy import LLMJudgeCompletionAccuracy
from eval_framework.shared.types import Completion, RawCompletion
from template_formatting.formatter import Message, Role


def test_llm_judge_completion_accuracy() -> None:
    response = Completion(
        id=0,
        subject="test",
        ground_truth="Berlin",
        prompt="test",
        prompt_sequence_positions=None,
        messages=[
            Message(role=Role.SYSTEM, content="You are a helpful AI agent."),
            Message(role=Role.USER, content="Can you say what's the capital of Germany?"),
        ],
        completion="Sure! The capital is Berlin. I hope this helps!",
        raw_completion="Sure! The capital is Berlin. I hope this helps!",
        raw_completion_sequence_positions=None,
    )

    llm = Mock(spec=BaseLLM)
    llm.generate_from_messages.return_value = [
        RawCompletion(
            prompt="prompt",
            completion=json.dumps({"answer_is_correct": True}),
            prompt_sequence_positions=None,
            completion_sequence_positions=None,
        )
    ]

    metric = LLMJudgeCompletionAccuracy(llm)
    results = metric.calculate(response)
    assert len(results) == 1
    assert results[0].value == 1.0
    assert results[0].llm_judge_prompt == "prompt"
    assert results[0].llm_judge_response

    response_bad = Completion(**response.model_dump(exclude={"ground_truth"}), ground_truth="Bonn.")

    llm = Mock(spec=BaseLLM)
    llm.generate_from_messages.return_value = [
        RawCompletion(
            prompt="prompt",
            completion=json.dumps({"answer_is_correct": False}),
            prompt_sequence_positions=None,
            completion_sequence_positions=None,
        )
    ]

    metric = LLMJudgeCompletionAccuracy(llm)
    results = metric.calculate(response_bad)
    assert len(results) == 1
    assert results[0].value == 0.0
    assert results[0].llm_judge_prompt == "prompt"
    assert results[0].llm_judge_response

    llm = Mock(spec=BaseLLM)
    llm.generate_from_messages.return_value = [
        RawCompletion(
            prompt="prompt",
            completion=json.dumps({"weird_json": "n/a"}),
            prompt_sequence_positions=None,
            completion_sequence_positions=None,
        )
    ]

    metric = LLMJudgeCompletionAccuracy(llm)
    results = metric.calculate(response)
    assert len(results) == 1
    assert results[0].value is None
    assert results[0].llm_judge_prompt == "prompt"
    assert results[0].llm_judge_response
