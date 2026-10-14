"""Specification of the Multi-IF benchmark.

The first turn is an ordinary 0-shot sample; later turns are asked during generation, each with the model's earlier
answers in context. ``test_formatter_hash`` pins the first turn against the real data.
"""

import json
from collections.abc import Sequence
from typing import Any, override

import pytest

from eval_framework.benchmarks.multi_if import MULTI_IF_BENCHMARKS, multi_if
from eval_framework.contract import Benchmark, Eval, Sample
from eval_framework.llm.base import BaseLLM
from eval_framework.metrics.completion.multi_if import MultiIFMetricContext
from eval_framework.shared.types import RawCompletion, RawLoglikelihood
from template_formatting.formatter import (
    BaseFormatter,
    ConcatFormatter,
    Llama3Formatter,
    Message,
    NoStripConcatFormatter,
    Role,
)
from tests.tests_eval_framework.benchmarks.utils import DatasetStub, assert_benchmark_formatter_hash, first_sample


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter, NoStripConcatFormatter])
@pytest.mark.parametrize("benchmark", MULTI_IF_BENCHMARKS, ids=lambda b: b.id())
def test_formatter_hash(benchmark: Benchmark, formatter_cls: type[BaseFormatter]) -> None:
    assert_benchmark_formatter_hash(benchmark, formatter_cls, num_fewshot=0)  # Multi-IF is 0-shot only


def _row(key: str, prompts: list[str]) -> dict[str, Any]:
    row: dict[str, Any] = {"key": key, "language": "English"}
    for turn in range(1, 4):
        if turn <= len(prompts):
            row[f"turn_{turn}_prompt"] = json.dumps({"role": "user", "content": prompts[turn - 1]})
            row[f"turn_{turn}_instruction_id_list"] = json.dumps(["punctuation:no_comma"] * turn)
            row[f"turn_{turn}_kwargs"] = json.dumps(["{}"] * turn)
        else:
            row[f"turn_{turn}_prompt"] = row[f"turn_{turn}_instruction_id_list"] = row[f"turn_{turn}_kwargs"] = ""
    return row


class _RecordingLlm(BaseLLM):
    """Answers with ``answer <n>`` for the n-th call it receives for a conversation and records every prompt."""

    def __init__(self) -> None:
        self.prompts: list[list[Message]] = []

    @override
    def generate_from_messages(
        self,
        messages: list[Sequence[Message]],
        stop_sequences: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> list[RawCompletion]:
        self.prompts += [list(m) for m in messages]
        return [
            RawCompletion(
                prompt="", prompt_num_tokens=0, completion=f"answer {len(m) // 2 + 1}", completion_num_tokens=0
            )
            for m in messages
        ]

    @override
    def logprobs(self, samples: list[Sample]) -> list[RawLoglikelihood]:
        raise NotImplementedError


def _english_eval(rows: list[dict[str, Any]]) -> tuple[Eval, list[Sample]]:
    evaluation = multi_if(dataset=DatasetStub({"train": rows})).create(
        num_fewshot=0, custom_subjects=["English"], custom_hf_revision=None, seed=42
    )
    return evaluation, list(evaluation.iterate_samples())


def test_first_turn_is_the_prompt_and_later_turns_live_in_the_context() -> None:
    # Given
    benchmark = multi_if(dataset=DatasetStub({"train": [_row("1:1:en", ["Write a poem.", "Shorter.", "In French."])]}))

    # When
    sample = first_sample(benchmark, num_fewshot=0, custom_subjects=["English"])

    # Then
    assert sample.messages == [Message(role=Role.USER, content="Write a poem.")]
    assert sample.ground_truth is None
    assert sample.context == MultiIFMetricContext(
        key="1:1:en",
        language="English",
        turn_prompts=["Write a poem.", "Shorter.", "In French."],
        turn_instruction_id_lists=[["punctuation:no_comma"] * n for n in (1, 2, 3)],
        turn_kwargs=[[{}] * n for n in (1, 2, 3)],
    )


def test_each_turn_sees_the_earlier_answers() -> None:
    # Given
    evaluation, samples = _english_eval([_row("1:1:en", ["Write a poem.", "Shorter.", "In French."])])
    llm = _RecordingLlm()

    # When
    [completion] = evaluation.generate_completions(llm, samples)

    # Then
    assert llm.prompts[-1] == [
        Message(role=Role.USER, content="Write a poem."),
        Message(role=Role.ASSISTANT, content="answer 1"),
        Message(role=Role.USER, content="Shorter."),
        Message(role=Role.ASSISTANT, content="answer 2"),
        Message(role=Role.USER, content="In French."),
    ]
    assert completion.messages == llm.prompts[-1] + [Message(role=Role.ASSISTANT, content="answer 3")]


def test_shorter_conversations_stop_after_their_last_turn() -> None:
    # Given a two-turn and a three-turn conversation in one batch
    evaluation, samples = _english_eval([_row("1:1:en", ["A.", "B."]), _row("2:1:en", ["C.", "D.", "E."])])
    llm = _RecordingLlm()

    # When
    completions = evaluation.generate_completions(llm, samples)

    # Then the model is asked five turns in total, and each conversation ends after its own last turn
    assert len(llm.prompts) == 5
    assert {c.messages[0].content: len(c.messages) for c in completions if c.messages} == {"A.": 4, "C.": 6}
