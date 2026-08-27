"""Specification of the German GPQA (EllaMind) tasks.

Each spec test builds the real benchmark (via its ``gpqa_ellamind_*_de`` constructor) over a fictional
dataset and asserts the assembled messages, ground truth, and scored completions. This is an Open
Source codebase, and the actual dataset should not leak in order to prevent it to become part of the
training data itself. Therfore it is important for the data to be fictional.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from eval_framework.benchmarks.gpqa_ellamind import (
    gpqa_ellamind_bpb_de,
    gpqa_ellamind_cloze_de,
    gpqa_ellamind_diamond_bpb_de,
    gpqa_ellamind_diamond_cloze_de,
    gpqa_ellamind_diamond_mc_de,
    gpqa_ellamind_mc_de,
)
from eval_framework.contract import Benchmark
from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_names import register_gpqa_ellamind_tasks
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter, Message, Role
from tests.tests_eval_framework.benchmarks.utils import DatasetStub, first_sample
from tests.tests_eval_framework.tasks.benchmarks.utils import run_formatter_hash_test

# Registry for this test suite only holding gpqa_ellamind tasks
_gpqa_ellamind_registry = Registry()
register_gpqa_ellamind_tasks(registry=_gpqa_ellamind_registry)


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter])
@pytest.mark.parametrize("task_name", _gpqa_ellamind_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, registry=_gpqa_ellamind_registry)


# ---------------------------------------------------------------------------
# Prompt spec: build the real benchmark over fictional rows, assert the assembled messages
# ---------------------------------------------------------------------------

# Fictional rows in the GPQA format (NOT real dataset examples). Choices are shuffled deterministically
# (seed: question + correct_answer). Both are diamond rows, so they survive the diamond variants' filter
# and render identically to the full variants.
_EVAL_ROW: dict[str, Any] = {
    "question": "Was ist die SI-Einheit des elektrischen Widerstands?",
    "correct_answer": "Ohm",
    "incorrect_answers": ["Volt", "Ampere", "Watt"],
    "is_diamond": True,
}
_FEWSHOT_ROW: dict[str, Any] = {
    "question": "Was ist die SI-Einheit der Temperatur?",
    "correct_answer": "Kelvin",
    "incorrect_answers": ["Celsius", "Fahrenheit", "Joule"],
    "is_diamond": True,
}

_EVAL_Q = "Frage: Was ist die SI-Einheit des elektrischen Widerstands?"
_FEWSHOT_Q = "Frage: Was ist die SI-Einheit der Temperatur?"


@dataclass(frozen=True)
class _ExpectedPrompt:
    messages: list[Message]
    ground_truth: str
    possible_completions: list[str]


# --- Zero-shot ---
_MC = _ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content=f"{_EVAL_Q}\nA. Ohm\nB. Volt\nC. Watt\nD. Ampere\n"),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    ground_truth=" A",
    possible_completions=[" A", " B", " C", " D"],
)
# Cloze/BPB show no options, so the assembled messages are identical; only the scored completions differ.
_CLOZE_MESSAGES = [
    Message(role=Role.USER, content=f"{_EVAL_Q}\n"),
    Message(role=Role.ASSISTANT, content="Antwort:"),
]
_CLOZE = _ExpectedPrompt(
    messages=_CLOZE_MESSAGES,
    ground_truth=" Ohm",
    possible_completions=[" Ohm", " Volt", " Watt", " Ampere"],
)
_BPB = _ExpectedPrompt(
    messages=_CLOZE_MESSAGES,
    ground_truth=" Ohm",
    possible_completions=[" Ohm"],  # BPB scores only the gold continuation
)


@pytest.mark.parametrize(
    "make_benchmark, expected",
    [
        pytest.param(gpqa_ellamind_mc_de, _MC, id="mc"),
        pytest.param(gpqa_ellamind_diamond_mc_de, _MC, id="diamond_mc"),
        pytest.param(gpqa_ellamind_cloze_de, _CLOZE, id="cloze"),
        pytest.param(gpqa_ellamind_diamond_cloze_de, _CLOZE, id="diamond_cloze"),
        pytest.param(gpqa_ellamind_bpb_de, _BPB, id="bpb"),
        pytest.param(gpqa_ellamind_diamond_bpb_de, _BPB, id="diamond_bpb"),
    ],
)
def test_gpqa_zeroshot_prompt(make_benchmark: Callable[..., Benchmark], expected: _ExpectedPrompt) -> None:
    benchmark = make_benchmark(dataset=DatasetStub({"train": [_EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0)
    assert sample.messages == expected.messages
    assert sample.ground_truth == expected.ground_truth
    assert sample.possible_completions == expected.possible_completions


# --- One-shot: fewshot row rendered with its answer, then the eval row's own zero-shot prompt ---
_MC_FEWSHOT_MESSAGES = [
    Message(role=Role.USER, content=f"{_FEWSHOT_Q}\nA. Celsius\nB. Fahrenheit\nC. Joule\nD. Kelvin\n"),
    Message(role=Role.ASSISTANT, content="Antwort: D"),
]
_CLOZE_FEWSHOT_MESSAGES = [
    Message(role=Role.USER, content=f"{_FEWSHOT_Q}\n"),
    Message(role=Role.ASSISTANT, content="Antwort: Kelvin"),
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
        pytest.param(gpqa_ellamind_mc_de, _oneshot(_MC_FEWSHOT_MESSAGES, _MC), id="mc"),
        pytest.param(gpqa_ellamind_diamond_mc_de, _oneshot(_MC_FEWSHOT_MESSAGES, _MC), id="diamond_mc"),
        pytest.param(gpqa_ellamind_cloze_de, _oneshot(_CLOZE_FEWSHOT_MESSAGES, _CLOZE), id="cloze"),
        pytest.param(gpqa_ellamind_diamond_cloze_de, _oneshot(_CLOZE_FEWSHOT_MESSAGES, _CLOZE), id="diamond_cloze"),
        pytest.param(gpqa_ellamind_bpb_de, _oneshot(_CLOZE_FEWSHOT_MESSAGES, _BPB), id="bpb"),
        pytest.param(gpqa_ellamind_diamond_bpb_de, _oneshot(_CLOZE_FEWSHOT_MESSAGES, _BPB), id="diamond_bpb"),
    ],
)
def test_gpqa_oneshot_prompt(make_benchmark: Callable[..., Benchmark], expected: _ExpectedPrompt) -> None:
    benchmark = make_benchmark(dataset=DatasetStub({"train": [_FEWSHOT_ROW, _EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=1)
    assert sample.messages == expected.messages
    assert sample.ground_truth == expected.ground_truth
    assert sample.possible_completions == expected.possible_completions


# ---------------------------------------------------------------------------
# Diamond subset: the diamond variant keeps only is_diamond rows; the full variant keeps them all
# ---------------------------------------------------------------------------
def test_gpqa_diamond_variant_keeps_only_diamond_rows() -> None:
    # Given a dataset mixing diamond and non-diamond rows
    rows: list[dict[str, Any]] = [
        {"question": "Q1", "correct_answer": "A", "incorrect_answers": ["x", "y", "z"], "is_diamond": True},
        {"question": "Q2", "correct_answer": "A", "incorrect_answers": ["x", "y", "z"], "is_diamond": False},
        {"question": "Q3", "correct_answer": "A", "incorrect_answers": ["x", "y", "z"], "is_diamond": True},
    ]
    diamond = gpqa_ellamind_diamond_mc_de(dataset=DatasetStub({"train": rows}))
    full = gpqa_ellamind_mc_de(dataset=DatasetStub({"train": rows}))

    # When we assemble all samples for each
    diamond_samples = list(diamond.create(0, None, None, seed=42).iterate_samples())
    full_samples = list(full.create(0, None, None, seed=42).iterate_samples())

    # Then the diamond variant drops the non-diamond row (Q2); the full variant keeps all three
    assert len(diamond_samples) == 2
    assert all("Q2" not in sample.messages[0].content for sample in diamond_samples)
    assert len(full_samples) == 3
