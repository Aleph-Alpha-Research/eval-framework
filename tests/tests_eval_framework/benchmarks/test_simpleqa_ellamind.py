"""Specification of the German SimpleQA (verified, EllaMind) tasks.

Each spec test builds the real benchmark (via its ``simpleqa_ellamind_*_de`` constructor) over a fictional
dataset and asserts the assembled messages, ground truth, and scored completions — so this file reads as
SimpleQA's prompt spec, with ``composed.py`` an implementation detail. The rows are fictional so this open
source codebase does not leak the real dataset into training data. ``test_formatter_hash`` separately pins
the real benchmarks against the real HuggingFace data.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from eval_framework.benchmarks.simpleqa_ellamind import (
    simpleqa_ellamind_bpb_de,
    simpleqa_ellamind_cloze_easy_de,
    simpleqa_ellamind_cloze_hard_de,
    simpleqa_ellamind_mc_easy_de,
    simpleqa_ellamind_mc_hard_de,
)
from eval_framework.contract import Benchmark
from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_names import register_simpleqa_ellamind_tasks
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter, Message, Role
from tests.tests_eval_framework.benchmarks.utils import DatasetStub, first_sample
from tests.tests_eval_framework.tasks.benchmarks.utils import run_formatter_hash_test

# Registry for this test suite only holding simpleqa_ellamind tasks
_simpleqa_ellamind_registry = Registry()
register_simpleqa_ellamind_tasks(registry=_simpleqa_ellamind_registry)


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter])
@pytest.mark.parametrize("task_name", _simpleqa_ellamind_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, registry=_simpleqa_ellamind_registry)


# ---------------------------------------------------------------------------
# Prompt spec: build the real benchmark over fictional rows, assert the assembled messages
# ---------------------------------------------------------------------------

# Fictional rows in the SimpleQA format (NOT real dataset examples). Choices are shuffled deterministically
# (seed: question + answer).
_EVAL_ROW: dict[str, Any] = {
    "question": "Welches Jahr haben?",
    "answer": "2026",
    "easy_distractors": ["1954", "1974", "1990"],
    "hard_distractors": ["2024", "2025", "2027"],
}
_FEWSHOT_ROW: dict[str, Any] = {
    "question": "Was ist die Hauptstadt von Frankreich?",
    "answer": "Paris",
    "easy_distractors": ["London", "Berlin", "Madrid"],
    "hard_distractors": ["Lyon", "Bordeaux", "Marseille"],
}

_EVAL_Q = "Frage: Welches Jahr haben?"


@dataclass(frozen=True)
class _ExpectedPrompt:
    messages: list[Message]
    ground_truth: str
    possible_completions: list[str]


# --- Zero-shot ---
_MC_EASY = _ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content=f"{_EVAL_Q}\nA. 1990\nB. 1954\nC. 1974\nD. 2026\n"),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    ground_truth=" D",
    possible_completions=[" A", " B", " C", " D"],
)
_MC_HARD = _ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content=f"{_EVAL_Q}\nA. 2027\nB. 2024\nC. 2025\nD. 2026\n"),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    ground_truth=" D",
    possible_completions=[" A", " B", " C", " D"],
)
# Cloze/BPB show no options, so the assembled messages are identical; only the scored completions differ.
_CLOZE_MESSAGES = [
    Message(role=Role.USER, content=f"{_EVAL_Q}\n"),
    Message(role=Role.ASSISTANT, content="Antwort:"),
]
_CLOZE_EASY = _ExpectedPrompt(
    messages=_CLOZE_MESSAGES,
    ground_truth=" 2026",
    possible_completions=[" 1990", " 1954", " 1974", " 2026"],
)
_CLOZE_HARD = _ExpectedPrompt(
    messages=_CLOZE_MESSAGES,
    ground_truth=" 2026",
    possible_completions=[" 2027", " 2024", " 2025", " 2026"],
)
_BPB = _ExpectedPrompt(
    messages=_CLOZE_MESSAGES,
    ground_truth=" 2026",
    possible_completions=[" 2026"],  # BPB scores only the gold continuation
)


@pytest.mark.parametrize(
    "make_benchmark, expected",
    [
        pytest.param(simpleqa_ellamind_mc_easy_de, _MC_EASY, id="mc_easy"),
        pytest.param(simpleqa_ellamind_mc_hard_de, _MC_HARD, id="mc_hard"),
        pytest.param(simpleqa_ellamind_cloze_easy_de, _CLOZE_EASY, id="cloze_easy"),
        pytest.param(simpleqa_ellamind_cloze_hard_de, _CLOZE_HARD, id="cloze_hard"),
        pytest.param(simpleqa_ellamind_bpb_de, _BPB, id="bpb"),
    ],
)
def test_simpleqa_zeroshot_prompt(make_benchmark: Callable[..., Benchmark], expected: _ExpectedPrompt) -> None:
    benchmark = make_benchmark(dataset=DatasetStub({"eval": [_EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0)
    assert sample.messages == expected.messages
    assert sample.ground_truth == expected.ground_truth
    assert sample.possible_completions == expected.possible_completions


# --- One-shot: fewshot row rendered with its answer, then the eval row's own zero-shot prompt ---
_MC_EASY_FEWSHOT_MESSAGES = [
    Message(
        role=Role.USER,
        content="Frage: Was ist die Hauptstadt von Frankreich?\nA. Paris\nB. Berlin\nC. London\nD. Madrid\n",
    ),
    Message(role=Role.ASSISTANT, content="Antwort: A"),
]
_MC_HARD_FEWSHOT_MESSAGES = [
    Message(
        role=Role.USER,
        content="Frage: Was ist die Hauptstadt von Frankreich?\nA. Paris\nB. Bordeaux\nC. Lyon\nD. Marseille\n",
    ),
    Message(role=Role.ASSISTANT, content="Antwort: A"),
]
_CLOZE_FEWSHOT_MESSAGES = [
    Message(role=Role.USER, content="Frage: Was ist die Hauptstadt von Frankreich?\n"),
    Message(role=Role.ASSISTANT, content="Antwort: Paris"),
]


def _oneshot(fewshot_messages: list[Message], eval_expected: _ExpectedPrompt) -> _ExpectedPrompt:
    return _ExpectedPrompt(
        messages=[*fewshot_messages, *eval_expected.messages],
        ground_truth=eval_expected.ground_truth,
        possible_completions=eval_expected.possible_completions,
    )


@pytest.mark.parametrize(
    "make_benchmark, expected",
    [
        pytest.param(simpleqa_ellamind_mc_easy_de, _oneshot(_MC_EASY_FEWSHOT_MESSAGES, _MC_EASY), id="mc_easy"),
        pytest.param(simpleqa_ellamind_mc_hard_de, _oneshot(_MC_HARD_FEWSHOT_MESSAGES, _MC_HARD), id="mc_hard"),
        pytest.param(simpleqa_ellamind_cloze_easy_de, _oneshot(_CLOZE_FEWSHOT_MESSAGES, _CLOZE_EASY), id="cloze_easy"),
        pytest.param(simpleqa_ellamind_cloze_hard_de, _oneshot(_CLOZE_FEWSHOT_MESSAGES, _CLOZE_HARD), id="cloze_hard"),
        pytest.param(simpleqa_ellamind_bpb_de, _oneshot(_CLOZE_FEWSHOT_MESSAGES, _BPB), id="bpb"),
    ],
)
def test_simpleqa_oneshot_prompt(make_benchmark: Callable[..., Benchmark], expected: _ExpectedPrompt) -> None:
    benchmark = make_benchmark(dataset=DatasetStub({"eval": [_FEWSHOT_ROW, _EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=1)
    assert sample.messages == expected.messages
    assert sample.ground_truth == expected.ground_truth
    assert sample.possible_completions == expected.possible_completions
