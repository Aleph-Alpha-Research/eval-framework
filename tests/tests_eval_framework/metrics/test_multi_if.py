from typing import Any

import pytest

from eval_framework.metrics.completion.multi_if import MultiIFMetric, MultiIFMetricContext
from eval_framework.shared.types import Completion, Error
from template_formatting.formatter import Message, Role


def _context(turns: list[tuple[list[str], list[dict[str, Any]]]]) -> MultiIFMetricContext:
    return MultiIFMetricContext(
        key="1:1:en",
        language="English",
        turn_prompts=[f"prompt {i}" for i in range(1, len(turns) + 1)],
        turn_instruction_id_lists=[ids for ids, _ in turns],
        turn_kwargs=[kwargs for _, kwargs in turns],
    )


def _completion(context: MultiIFMetricContext, answers: list[str], error: Error | None = None) -> Completion:
    messages = []
    for prompt, answer in zip(context.turn_prompts, answers):
        messages += [Message(role=Role.USER, content=prompt), Message(role=Role.ASSISTANT, content=answer)]
    return Completion(
        id=0,
        subject="English",
        ground_truth=None,
        prompt="",
        prompt_num_tokens=None,
        messages=messages,
        completion=answers[-1],
        raw_completion=answers[-1],
        raw_completion_num_tokens=None,
        context=context,
        error=error,
    )


def _values(results: list, name: str) -> list[float | None]:
    return [result.value for result in results if result.metric_name == name]


def test_each_turn_is_graded_against_its_own_instructions() -> None:
    # Given a turn 1 asking for no commas and a turn 2 adding lowercase on top
    context = _context(
        [
            (["punctuation:no_comma"], [{}]),
            (["punctuation:no_comma", "change_case:english_lowercase"], [{}, {}]),
        ]
    )
    # ... where turn 1 follows its instruction and turn 2 breaks the new one
    completion = _completion(context, ["The sea is calm today.", "The sea is still calm today."])

    # When
    results = MultiIFMetric().calculate(completion)

    # Then
    assert _values(results, "Multi-IF Turn 1/prompt_level_strict_acc") == [1.0]
    assert _values(results, "Multi-IF Turn 1/inst_level_strict_acc") == [1.0]
    assert _values(results, "Multi-IF Turn 2/prompt_level_strict_acc") == [0.0]
    assert _values(results, "Multi-IF Turn 2/inst_level_strict_acc") == [1.0, 0.0]


def test_loose_accuracy_ignores_markdown_emphasis() -> None:
    # Given an answer whose closing phrase is wrapped in bold markers
    context = _context([(["startend:end_checker"], [{"end_phrase": "Any other questions?"}])])
    completion = _completion(context, ["Here you go. **Any other questions?**"])

    # When
    results = MultiIFMetric().calculate(completion)

    # Then
    assert _values(results, "Multi-IF Turn 1/prompt_level_strict_acc") == [0.0]
    assert _values(results, "Multi-IF Turn 1/prompt_level_loose_acc") == [1.0]


@pytest.mark.parametrize(
    "instruction_id, kwargs, answer",
    [
        # 14 CJK characters count as 14 words; whitespace splitting would count one.
        (
            "length_constraints:number_words",
            {"relation": "at least", "num_words": 10},
            "今天天气很好我们去公园散步吧",
        ),
        # Three sentences ended by the danda; the English tokenizer sees one.
        (
            "length_constraints:number_sentences",
            {"relation": "at least", "num_sentences": 3},
            "आज मौसम बहुत अच्छा है। हम पार्क में टहलने जाएंगे। फिर हम घर लौटेंगे।",
        ),
        ("startend:quotation", {}, "“这是一个被引号包裹的回答”"),
    ],
    ids=["chinese_word_count", "hindi_sentence_count", "curly_quotes"],
)
def test_language_aware_checkers_accept_non_latin_answers(
    instruction_id: str, kwargs: dict[str, Any], answer: str
) -> None:
    # Given
    completion = _completion(_context([([instruction_id], [kwargs])]), [answer])

    # When
    results = MultiIFMetric().calculate(completion)

    # Then
    assert _values(results, "Multi-IF Turn 1/prompt_level_strict_acc") == [1.0]


def test_failed_conversation_reports_errors_for_every_turn() -> None:
    # Given a two-turn conversation that failed during generation
    context = _context([(["punctuation:no_comma"], [{}]), (["punctuation:no_comma"], [{}])])
    error = Error(error_class="RuntimeError", message="backend down", traceback="")
    completion = _completion(context, ["", ""], error=error)

    # When
    results = MultiIFMetric().calculate(completion)

    # Then
    assert {result.metric_name for result in results} == {
        "Multi-IF Turn 1/prompt_level_strict_acc",
        "Multi-IF Turn 1/prompt_level_loose_acc",
        "Multi-IF Turn 2/prompt_level_strict_acc",
        "Multi-IF Turn 2/prompt_level_loose_acc",
    }
    assert all(result.value is None and result.error == error for result in results)
