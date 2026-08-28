"""Specification of the German HLE (Humanity's Last Exam, EllaMind) tasks.

Each spec test builds the real benchmark (via its ``hle_ellamind_*_de`` constructor) over a fictional
dataset and asserts the assembled messages, ground truth, and scored completions — so this file reads as
HLE's prompt spec, with ``composed.py`` an implementation detail. The rows are fictional so this open
source codebase does not leak the real dataset into training data. NATIVE variants render identically to
their full counterparts; they differ only by restricting the dataset to natively-multiple-choice items
(``answer_type == "multipleChoice"``). ``test_formatter_hash`` separately pins the real benchmarks against
the real HuggingFace data.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from eval_framework.benchmarks.hle_ellamind import (
    hle_ellamind_bpb_de,
    hle_ellamind_cloze_de,
    hle_ellamind_cloze_native_de,
    hle_ellamind_mc_de,
    hle_ellamind_mc_native_de,
)
from eval_framework.contract import Benchmark
from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_names import register_hle_ellamind_tasks
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter, Message, Role
from tests.tests_eval_framework.benchmarks.utils import DatasetStub, first_sample
from tests.tests_eval_framework.tasks.benchmarks.utils import run_formatter_hash_test

# Registry for this test suite only holding hle_ellamind tasks
_hle_ellamind_registry = Registry()
register_hle_ellamind_tasks(registry=_hle_ellamind_registry)


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter])
@pytest.mark.parametrize("task_name", _hle_ellamind_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, registry=_hle_ellamind_registry)


# ---------------------------------------------------------------------------
# Prompt spec: build the real benchmark over fictional rows, assert the assembled messages
# ---------------------------------------------------------------------------

# Fictional rows in the HLE format (NOT real dataset examples). Choices are shuffled deterministically
# (seed: question + correct_answer). Both are multipleChoice rows, so they survive the NATIVE filter and
# render identically to the full variants.
_EVAL_ROW: dict[str, Any] = {
    "question": "Was ist die Antwort auf die Frage nach dem Leben, dem Universum und allem?",
    "correct_answer": "42",
    "incorrect_answers": ["41", "43", "44"],
    "answer_type": "multipleChoice",
}
_FEWSHOT_ROW: dict[str, Any] = {
    "question": "Was ist die Summe der Innenwinkel eines Dreiecks?",
    "correct_answer": "180",
    "incorrect_answers": ["90", "270", "360"],
    "answer_type": "multipleChoice",
}

_EVAL_Q = "Frage: Was ist die Antwort auf die Frage nach dem Leben, dem Universum und allem?"
_FEWSHOT_Q = "Frage: Was ist die Summe der Innenwinkel eines Dreiecks?"


@dataclass(frozen=True)
class _ExpectedPrompt:
    messages: list[Message]
    ground_truth: str
    possible_completions: list[str]


# --- Zero-shot ---
_MC = _ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content=f"{_EVAL_Q}\nA. 44\nB. 43\nC. 41\nD. 42\n"),
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
_CLOZE = _ExpectedPrompt(
    messages=_CLOZE_MESSAGES,
    ground_truth=" 42",
    possible_completions=[" 44", " 43", " 41", " 42"],
)
_BPB = _ExpectedPrompt(
    messages=_CLOZE_MESSAGES,
    ground_truth=" 42",
    possible_completions=[" 42"],  # BPB scores only the gold continuation
)


@pytest.mark.parametrize(
    "make_benchmark, expected",
    [
        pytest.param(hle_ellamind_mc_de, _MC, id="mc"),
        pytest.param(hle_ellamind_mc_native_de, _MC, id="mc_native"),
        pytest.param(hle_ellamind_cloze_de, _CLOZE, id="cloze"),
        pytest.param(hle_ellamind_cloze_native_de, _CLOZE, id="cloze_native"),
        pytest.param(hle_ellamind_bpb_de, _BPB, id="bpb"),
    ],
)
def test_hle_zeroshot_prompt(make_benchmark: Callable[..., Benchmark], expected: _ExpectedPrompt) -> None:
    benchmark = make_benchmark(dataset=DatasetStub({"test": [_EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0)
    assert sample.messages == expected.messages
    assert sample.ground_truth == expected.ground_truth
    assert sample.possible_completions == expected.possible_completions


# --- One-shot: fewshot row rendered with its answer, then the eval row's own zero-shot prompt ---
_MC_FEWSHOT_MESSAGES = [
    Message(role=Role.USER, content=f"{_FEWSHOT_Q}\nA. 270\nB. 180\nC. 360\nD. 90\n"),
    Message(role=Role.ASSISTANT, content="Antwort: B"),
]
_CLOZE_FEWSHOT_MESSAGES = [
    Message(role=Role.USER, content=f"{_FEWSHOT_Q}\n"),
    Message(role=Role.ASSISTANT, content="Antwort: 180"),
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
        pytest.param(hle_ellamind_mc_de, _oneshot(_MC_FEWSHOT_MESSAGES, _MC), id="mc"),
        pytest.param(hle_ellamind_mc_native_de, _oneshot(_MC_FEWSHOT_MESSAGES, _MC), id="mc_native"),
        pytest.param(hle_ellamind_cloze_de, _oneshot(_CLOZE_FEWSHOT_MESSAGES, _CLOZE), id="cloze"),
        pytest.param(hle_ellamind_cloze_native_de, _oneshot(_CLOZE_FEWSHOT_MESSAGES, _CLOZE), id="cloze_native"),
        pytest.param(hle_ellamind_bpb_de, _oneshot(_CLOZE_FEWSHOT_MESSAGES, _BPB), id="bpb"),
    ],
)
def test_hle_oneshot_prompt(make_benchmark: Callable[..., Benchmark], expected: _ExpectedPrompt) -> None:
    benchmark = make_benchmark(dataset=DatasetStub({"test": [_FEWSHOT_ROW, _EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=1)
    assert sample.messages == expected.messages
    assert sample.ground_truth == expected.ground_truth
    assert sample.possible_completions == expected.possible_completions


# ---------------------------------------------------------------------------
# NATIVE subset: the native variant keeps only answer_type == "multipleChoice" rows; the full variant keeps all
# ---------------------------------------------------------------------------
def test_hle_native_variant_keeps_only_multiple_choice_rows() -> None:
    # Given a dataset mixing multipleChoice and other answer types
    rows: list[dict[str, Any]] = [
        {
            "question": "Q1",
            "correct_answer": "a",
            "incorrect_answers": ["x", "y", "z"],
            "answer_type": "multipleChoice",
        },
        {"question": "Q2", "correct_answer": "a", "incorrect_answers": ["x", "y", "z"], "answer_type": "exactMatch"},
        {
            "question": "Q3",
            "correct_answer": "a",
            "incorrect_answers": ["x", "y", "z"],
            "answer_type": "multipleChoice",
        },
    ]
    native = hle_ellamind_mc_native_de(dataset=DatasetStub({"test": rows}))
    full = hle_ellamind_mc_de(dataset=DatasetStub({"test": rows}))

    # When we assemble all samples for each
    native_samples = list(native.create(0, None, None, seed=42).iterate_samples())
    full_samples = list(full.create(0, None, None, seed=42).iterate_samples())

    # Then the native variant drops the non-multipleChoice row (Q2); the full variant keeps all three
    assert len(native_samples) == 2
    assert all("Q2" not in sample.messages[0].content for sample in native_samples)
    assert len(full_samples) == 3
