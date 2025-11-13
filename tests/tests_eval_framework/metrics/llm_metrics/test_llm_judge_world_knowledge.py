import json
from unittest.mock import Mock

from eval_framework.llm.base import BaseLLM
from eval_framework.metrics.llm.llm_judge_world_knowledge import LLMJudgeWorldKnowledge
from eval_framework.shared.types import Completion, RawCompletion
from template_formatting.formatter import Message, Role


def test_llm_judge_world_knowledge() -> None:
    response = Completion(
        id=0,
        subject="test",
        ground_truth=None,
        prompt="test",
        prompt_sequence_positions=None,
        messages=[
            Message(role=Role.SYSTEM, content="You are a helpful AI agent which can summarize texts"),
            Message(role=Role.USER, content="Berlin is a lovable city. I really enjoy the food there."),
        ],
        completion="Berlin is the capital city of Germany, it's lovable.",
        raw_completion="Berlin is the capital city of Germany, it's lovable.",
        raw_completion_sequence_positions=None,
    )

    llm = Mock(spec=BaseLLM)
    llm.generate_from_messages.return_value = [
        RawCompletion(
            prompt="prompt",
            completion=json.dumps(
                {
                    "contains_world_knowledge_thought_process": "n/a",
                    "contains_world_knowledge": True,
                }
            ),
            prompt_sequence_positions=None,
            completion_sequence_positions=None,
        )
    ]

    metric = LLMJudgeWorldKnowledge(llm)
    results = metric.calculate(response)
    assert len(results) == 1
    assert results[0].value == 1.0

    response_short = Completion(**response.model_dump(exclude={"completion"}), completion="I like Berlin.")

    llm = Mock(spec=BaseLLM)
    llm.generate_from_messages.return_value = [
        RawCompletion(
            prompt="prompt",
            completion=json.dumps(
                {
                    "contains_world_knowledge_thought_process": "n/a",
                    "contains_world_knowledge": False,
                }
            ),
            prompt_sequence_positions=None,
            completion_sequence_positions=None,
        )
    ]

    metric = LLMJudgeWorldKnowledge(llm)
    results = metric.calculate(response_short)
    assert len(results) == 1
    assert results[0].value == 0.0

    llm = Mock(spec=BaseLLM)
    llm.generate_from_messages.return_value = [
        RawCompletion(
            prompt="prompt",
            completion=json.dumps({"contains": "bad key"}),
            prompt_sequence_positions=None,
            completion_sequence_positions=None,
        )
    ]

    metric = LLMJudgeWorldKnowledge(llm)
    results = metric.calculate(response)
    assert len(results) == 1
    assert results[0].value is None
    assert results[0].llm_judge_prompt == "prompt"
    assert results[0].llm_judge_response
