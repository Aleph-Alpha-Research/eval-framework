import json
from unittest.mock import Mock

from eval_framework.llm.base import BaseLLM
from eval_framework.metrics.llm.llm_judge_conciseness import LLMJudgeConciseness
from eval_framework.shared.types import Completion, RawCompletion
from template_formatting.formatter import Message, Role


def test_llm_judge_consiseness() -> None:
    response = Completion(
        id=0,
        subject="test",
        ground_truth=None,
        prompt="test",
        prompt_sequence_positions=None,
        messages=[
            Message(role=Role.SYSTEM, content="You are a helpful AI agent."),
            Message(role=Role.USER, content="Can you say what's the capital of Germany?"),
        ],
        completion="Berlin, Germany’s capital, dates to the 13th century. Reminders of the city's turbulent ...",
        raw_completion="Berlin, Germany’s capital, dates to the 13th century. Reminders of the city's turbulent ...",
        raw_completion_sequence_positions=None,
    )

    llm = Mock(spec=BaseLLM)
    llm.generate_from_messages.return_value = [
        RawCompletion(
            prompt="prompt",
            completion=json.dumps({"thought_process": "n/a", "is_concise": False}),
            prompt_sequence_positions=None,
            completion_sequence_positions=None,
        )
    ]

    metric = LLMJudgeConciseness(llm)
    results = metric.calculate(response)
    assert len(results) == 1
    assert results[0].value == 0.0
    assert results[0].llm_judge_prompt == "prompt"
    assert results[0].llm_judge_response

    response_short = Completion(**response.model_dump(exclude={"completion"}), completion="Berlin.")

    llm = Mock(spec=BaseLLM)
    llm.generate_from_messages.return_value = [
        RawCompletion(
            prompt="prompt",
            completion=json.dumps({"thought_process": "n/a", "is_concise": True}),
            prompt_sequence_positions=None,
            completion_sequence_positions=None,
        )
    ]

    metric = LLMJudgeConciseness(llm)
    results = metric.calculate(response_short)
    assert len(results) == 1
    assert results[0].value == 1.0
    assert results[0].llm_judge_prompt == "prompt"
    assert results[0].llm_judge_response

    llm = Mock(spec=BaseLLM)
    llm.generate_from_messages.return_value = [
        RawCompletion(
            prompt="prompt",
            completion=json.dumps({"thought_process": "n/a"}),
            prompt_sequence_positions=None,
            completion_sequence_positions=None,
        )
    ]

    metric = LLMJudgeConciseness(llm)
    results = metric.calculate(response)
    assert len(results) == 1
    assert results[0].value is None
    assert results[0].llm_judge_prompt == "prompt"
    assert results[0].llm_judge_response
