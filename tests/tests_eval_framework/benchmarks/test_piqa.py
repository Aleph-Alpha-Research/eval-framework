"""Specification of the PIQA tasks.

Each spec test builds the real benchmark (via its ``piqa*`` constructor) over a fictional dataset and
asserts the assembled messages, ground truth, and scored completions — so this file reads as PIQA's
prompt spec, with ``composed.py`` an implementation detail. ``test_formatter_hash`` separately pins the
real benchmarks against the real HuggingFace data.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from eval_framework.benchmarks.piqa import _IDK_PREAMBLE, piqa, piqa_idk, piqa_olmes
from eval_framework.contract import Benchmark
from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_names import register_piqa_tasks
from template_formatting.formatter import (
    BaseFormatter,
    ConcatFormatter,
    Llama3Formatter,
    Message,
    NoStripConcatFormatter,
    Role,
)
from tests.tests_eval_framework.benchmarks.utils import DatasetStub, first_sample
from tests.tests_eval_framework.tasks.benchmarks.utils import run_formatter_hash_test

# Registry for this test suite only holding piqa tasks
_piqa_registry = Registry()
register_piqa_tasks(registry=_piqa_registry)


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter, NoStripConcatFormatter])
@pytest.mark.parametrize("task_name", _piqa_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, registry=_piqa_registry)


# ---------------------------------------------------------------------------
# Prompt spec: build the real benchmark over fictional rows, assert the assembled messages
# ---------------------------------------------------------------------------

# Fictional rows in the PIQA format (NOT real dataset examples): a goal with two candidate solutions,
# ``label`` selecting the correct one. Solutions are shown in fixed order (no shuffle).
_EVAL_ROW: dict[str, Any] = {
    "goal": "How do you dig a hole?",
    "sol1": "Use a shovel to remove the dirt.",
    "sol2": "Pour water on the ground until a hole forms.",
    "label": 0,
}
_FEWSHOT_ROW: dict[str, Any] = {
    "goal": "How do you light a candle?",
    "sol1": "Hold an ice cube against the wick.",
    "sol2": "Touch a lit match to the wick.",
    "label": 1,
}


@dataclass(frozen=True)
class _ExpectedPrompt:
    messages: list[Message]
    ground_truth: str
    possible_completions: list[str]


# --- Zero-shot ---
# Cloze: no options shown; scores the full solution text.
_CLOZE = _ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content="Question: How do you dig a hole?\n"),
        Message(role=Role.ASSISTANT, content="Answer:"),
    ],
    ground_truth=" Use a shovel to remove the dirt.",
    possible_completions=[" Use a shovel to remove the dirt.", " Pour water on the ground until a hole forms."],
)
# OLMES: solutions shown as space-prefixed lettered options; scores the letters.
_OLMES = _ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content="Goal: How do you dig a hole?\n"
            " A. Use a shovel to remove the dirt.\n B. Pour water on the ground until a hole forms.\n",
        ),
        Message(role=Role.ASSISTANT, content="Answer:"),
    ],
    ground_truth=" A",
    possible_completions=[" A", " B"],
)
# IDK: cloze plus an appended "I do not know" option, preceded once by the confidence preamble.
_IDK = _ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content=f"{_IDK_PREAMBLE}\n\nQuestion: How do you dig a hole?\n"),
        Message(role=Role.ASSISTANT, content="Answer:"),
    ],
    ground_truth=" Use a shovel to remove the dirt.",
    possible_completions=[
        " Use a shovel to remove the dirt.",
        " Pour water on the ground until a hole forms.",
        " I do not know",
    ],
)


@pytest.mark.parametrize(
    "make_benchmark, sample_split, expected",
    [
        pytest.param(piqa, "validation", _CLOZE, id="cloze"),
        pytest.param(piqa_olmes, "train", _OLMES, id="olmes"),
        pytest.param(piqa_idk, "validation", _IDK, id="idk"),
    ],
)
def test_piqa_zeroshot_prompt(
    make_benchmark: Callable[..., Benchmark], sample_split: str, expected: _ExpectedPrompt
) -> None:
    # Given the real PIQA benchmark over a single fictional row (on the variant's own sample split)
    benchmark = make_benchmark(dataset=DatasetStub({sample_split: [_EVAL_ROW]}))
    # When we assemble its first sample (zero-shot)
    sample = first_sample(benchmark, num_fewshot=0)
    # Then the messages, ground truth, and scored completions are exactly:
    assert sample.messages == expected.messages
    assert sample.ground_truth == expected.ground_truth
    assert sample.possible_completions == expected.possible_completions


# --- One-shot: the fewshot row rendered with its answer, then the eval row's own zero-shot prompt ---
_CLOZE_FEWSHOT_MESSAGES = [
    Message(role=Role.USER, content="Question: How do you light a candle?\n"),
    Message(role=Role.ASSISTANT, content="Answer: Touch a lit match to the wick."),
]
_OLMES_FEWSHOT_MESSAGES = [
    Message(
        role=Role.USER,
        content="Goal: How do you light a candle?\n"
        " A. Hold an ice cube against the wick.\n B. Touch a lit match to the wick.\n",
    ),
    Message(role=Role.ASSISTANT, content="Answer: B"),
]


def _oneshot(fewshot_messages: list[Message], eval_expected: _ExpectedPrompt) -> _ExpectedPrompt:
    return _ExpectedPrompt(
        messages=[*fewshot_messages, *eval_expected.messages],
        ground_truth=eval_expected.ground_truth,
        possible_completions=eval_expected.possible_completions,
    )


@pytest.mark.parametrize(
    "make_benchmark, splits, expected",
    [
        # Base/IDK draw few-shot from a separate split (validation/test); OLMES from its own (train).
        pytest.param(
            piqa,
            {"validation": [_EVAL_ROW], "test": [_FEWSHOT_ROW]},
            _oneshot(_CLOZE_FEWSHOT_MESSAGES, _CLOZE),
            id="cloze",
        ),
        pytest.param(
            piqa_olmes,
            {"train": [_FEWSHOT_ROW, _EVAL_ROW]},
            _oneshot(_OLMES_FEWSHOT_MESSAGES, _OLMES),
            id="olmes",
        ),
    ],
)
def test_piqa_oneshot_prompt(
    make_benchmark: Callable[..., Benchmark], splits: dict[str, list[dict[str, Any]]], expected: _ExpectedPrompt
) -> None:
    # Given the real PIQA benchmark over a fewshot row and the eval row
    benchmark = make_benchmark(dataset=DatasetStub(splits))
    # When we assemble its first sample (one-shot)
    sample = first_sample(benchmark, num_fewshot=1)
    # Then the fewshot example precedes the eval prompt, and ground truth / completions are the eval row's:
    assert sample.messages == expected.messages
    assert sample.ground_truth == expected.ground_truth
    assert sample.possible_completions == expected.possible_completions


def test_piqa_idk_oneshot_prepends_preamble_once_to_the_first_message() -> None:
    # Given the IDK benchmark with a fewshot row (separate split) and the eval row
    benchmark = piqa_idk(dataset=DatasetStub({"validation": [_EVAL_ROW], "test": [_FEWSHOT_ROW]}))
    # When we assemble its first sample (one-shot)
    sample = first_sample(benchmark, num_fewshot=1)
    # Then the preamble precedes only the first (fewshot) message, and the abstention option is scored
    assert sample.messages == [
        Message(role=Role.USER, content=f"{_IDK_PREAMBLE}\n\nQuestion: How do you light a candle?\n"),
        Message(role=Role.ASSISTANT, content="Answer: Touch a lit match to the wick."),
        Message(role=Role.USER, content="Question: How do you dig a hole?\n"),
        Message(role=Role.ASSISTANT, content="Answer:"),
    ]
    assert sample.possible_completions == [
        " Use a shovel to remove the dirt.",
        " Pour water on the ground until a hole forms.",
        " I do not know",
    ]
