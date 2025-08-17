import json
from unittest.mock import Mock

from eval_framework.llm.base import BaseLLM
from eval_framework.metrics.llm_metrics.llm_judge_instruction import LLMJudgeInstruction
from eval_framework.shared.types import Completion, RawCompletion
from template_formatting.formatter import Message, Role


def test_llm_judge_instruction() -> None:
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
        completion="Yes, Frankfurt is in Germany. You're so bad at geography.",
        raw_completion="Yes, Frankfurt is in Germany. You're so bad at geography.",
        raw_completion_sequence_positions=None,
    )

    llm = Mock(spec=BaseLLM)
    llm.generate_from_messages.return_value = [
        RawCompletion(
            prompt="prompt",
            completion=json.dumps(
                {
                    "criticism": "The response is not relevant to the user's question.",
                    "quality": "D",
                    "is_following_instruction": False,
                    "has_correct_grammar_and_spelling": True,
                    "is_contradicting_context": True,
                    "is_repeating": False,
                    "is_deceitful": False,
                    "is_harmful": True,
                }
            ),
            prompt_sequence_positions=None,
            completion_sequence_positions=None,
        )
    ]

    metric = LLMJudgeInstruction(llm)
    results = metric.calculate(response)
    assert len(results) == 7
    for result in results:
        assert result.llm_judge_prompt == "prompt"
        assert result.llm_judge_response
        match result.metric_name:
            case "Instruction Following/quality":
                assert result.value == (2 - 1) / 4  # (normalized)
            case "Instruction Following/is_following_instruction":
                assert result.value == 0.0
            case "Instruction Following/has_correct_grammar_and_spelling":
                assert result.value == 1.0
            case "Instruction Following/is_context_consistent":
                assert result.value == 0.0
            case "Instruction Following/is_not_repeating":
                assert result.value == 1.0
            case "Instruction Following/is_trustworthy":
                assert result.value == 1.0
            case "Instruction Following/is_safe":
                assert result.value == 0.0

    llm = Mock(spec=BaseLLM)
    llm.generate_from_messages.return_value = [
        RawCompletion(
            prompt="prompt",
            completion=json.dumps({"contains": "bad key"}),
            prompt_sequence_positions=None,
            completion_sequence_positions=None,
        )
    ]

    metric = LLMJudgeInstruction(llm)
    results = metric.calculate(response)
    assert len(results) == 7
    assert results[0].value is None
    assert results[0].llm_judge_prompt == "prompt"
    assert results[0].llm_judge_response
