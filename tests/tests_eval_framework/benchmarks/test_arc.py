"""Specification of the ARC tasks.

Each spec test builds the real benchmark (via its ``arc*`` constructor) over a fictional dataset and
asserts the assembled messages, ground truth, and scored completions — so this file reads as ARC's prompt
spec, with ``composed.py`` an implementation detail. ``test_formatter_hash`` separately pins the real
benchmarks against the real HuggingFace data.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from eval_framework.benchmarks.arc import _IDK_PREAMBLE, arc, arc_idk, arc_olmes
from eval_framework.contract import Benchmark
from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_names import register_arc_tasks
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

# Registry for this test suite only holding the composed arc tasks.
_arc_registry = Registry()
register_arc_tasks(registry=_arc_registry)


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter, NoStripConcatFormatter])
@pytest.mark.parametrize("task_name", _arc_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, registry=_arc_registry)


# ---------------------------------------------------------------------------
# Prompt spec: build the real benchmark over fictional rows, assert the assembled messages
# ---------------------------------------------------------------------------

# Fictional rows in the ARC format (NOT real dataset examples): a question, its labelled options, and the
# ``answerKey``. Options are shown in fixed order (no shuffle).
_EVAL_ROW: dict[str, Any] = {
    "question": "Which gas do plants release during photosynthesis?",
    "choices": {"text": ["Oxygen", "Carbon dioxide", "Nitrogen", "Hydrogen"], "label": ["A", "B", "C", "D"]},
    "answerKey": "A",
}
_FEWSHOT_ROW: dict[str, Any] = {
    "question": "What force pulls objects toward Earth?",
    "choices": {"text": ["Magnetism", "Gravity", "Friction", "Tension"], "label": ["A", "B", "C", "D"]},
    "answerKey": "B",
}


@dataclass(frozen=True)
class _ExpectedPrompt:
    messages: list[Message]
    ground_truth: str
    possible_completions: list[str]


# Cloze: no options shown; scores the full answer text.
_CLOZE = _ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content="Question: Which gas do plants release during photosynthesis?\n"),
        Message(role=Role.ASSISTANT, content="Answer:"),
    ],
    ground_truth=" Oxygen",
    possible_completions=[" Oxygen", " Carbon dioxide", " Nitrogen", " Hydrogen"],
)
# OLMES: options shown as space-prefixed lettered choices; scores the letters.
_OLMES = _ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content="Question: Which gas do plants release during photosynthesis?\n"
            " A. Oxygen\n B. Carbon dioxide\n C. Nitrogen\n D. Hydrogen\n",
        ),
        Message(role=Role.ASSISTANT, content="Answer:"),
    ],
    ground_truth=" A",
    possible_completions=[" A", " B", " C", " D"],
)
# IDK: cloze plus an appended "I do not know." option, preceded once by the confidence preamble.
_IDK = _ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content=f"{_IDK_PREAMBLE}\n\nQuestion: Which gas do plants release during photosynthesis?\n",
        ),
        Message(role=Role.ASSISTANT, content="Answer:"),
    ],
    ground_truth=" Oxygen",
    possible_completions=[" Oxygen", " Carbon dioxide", " Nitrogen", " Hydrogen", " I do not know."],
)


@pytest.mark.parametrize(
    "make_benchmark, expected",
    [
        pytest.param(arc, _CLOZE, id="cloze"),
        pytest.param(arc_olmes, _OLMES, id="olmes"),
        pytest.param(arc_idk, _IDK, id="idk"),
    ],
)
def test_arc_zeroshot_prompt(make_benchmark: Callable[..., Benchmark], expected: _ExpectedPrompt) -> None:
    # Given the real ARC benchmark over one fictional row on its sample split (test)
    benchmark = make_benchmark(dataset=DatasetStub({"test": [_EVAL_ROW]}))
    # When we assemble its first sample (zero-shot)
    sample = first_sample(benchmark, num_fewshot=0)
    # Then the messages, ground truth, and scored completions are exactly:
    assert sample.messages == expected.messages
    assert sample.ground_truth == expected.ground_truth
    assert sample.possible_completions == expected.possible_completions


# One-shot: the fewshot row rendered with its answer, then the eval row's own zero-shot prompt (few-shot
# is drawn from a separate split — train — so there is no leak-prevention subtlety here).
_CLOZE_FEWSHOT = [
    Message(role=Role.USER, content="Question: What force pulls objects toward Earth?\n"),
    Message(role=Role.ASSISTANT, content="Answer: Gravity"),
]
_OLMES_FEWSHOT = [
    Message(
        role=Role.USER,
        content="Question: What force pulls objects toward Earth?\n"
        " A. Magnetism\n B. Gravity\n C. Friction\n D. Tension\n",
    ),
    Message(role=Role.ASSISTANT, content="Answer: B"),
]


def _oneshot(fewshot: list[Message], eval_expected: _ExpectedPrompt) -> _ExpectedPrompt:
    return _ExpectedPrompt(
        messages=[*fewshot, *eval_expected.messages],
        ground_truth=eval_expected.ground_truth,
        possible_completions=eval_expected.possible_completions,
    )


@pytest.mark.parametrize(
    "make_benchmark, fewshot, eval_expected",
    [
        pytest.param(arc, _CLOZE_FEWSHOT, _CLOZE, id="cloze"),
        pytest.param(arc_olmes, _OLMES_FEWSHOT, _OLMES, id="olmes"),
    ],
)
def test_arc_oneshot_prompt(
    make_benchmark: Callable[..., Benchmark], fewshot: list[Message], eval_expected: _ExpectedPrompt
) -> None:
    # Given the real ARC benchmark over a fewshot row (train) and the eval row (test)
    benchmark = make_benchmark(dataset=DatasetStub({"test": [_EVAL_ROW], "train": [_FEWSHOT_ROW]}))
    # When we assemble its first sample (one-shot)
    sample = first_sample(benchmark, num_fewshot=1)
    # Then the fewshot example precedes the eval prompt, and ground truth / completions are the eval row's:
    expected = _oneshot(fewshot, eval_expected)
    assert sample.messages == expected.messages
    assert sample.ground_truth == expected.ground_truth
    assert sample.possible_completions == expected.possible_completions


def test_arc_idk_oneshot_prepends_preamble_once_to_the_first_message() -> None:
    # Given the IDK benchmark with a fewshot row (train) and the eval row (test)
    benchmark = arc_idk(dataset=DatasetStub({"test": [_EVAL_ROW], "train": [_FEWSHOT_ROW]}))
    # When we assemble its first sample (one-shot)
    sample = first_sample(benchmark, num_fewshot=1)
    # Then the preamble precedes only the first (fewshot) message, and the abstention option is scored
    assert sample.messages == [
        Message(role=Role.USER, content=f"{_IDK_PREAMBLE}\n\nQuestion: What force pulls objects toward Earth?\n"),
        Message(role=Role.ASSISTANT, content="Answer: Gravity"),
        Message(role=Role.USER, content="Question: Which gas do plants release during photosynthesis?\n"),
        Message(role=Role.ASSISTANT, content="Answer:"),
    ]
    assert sample.possible_completions == [
        " Oxygen",
        " Carbon dioxide",
        " Nitrogen",
        " Hydrogen",
        " I do not know.",
    ]
