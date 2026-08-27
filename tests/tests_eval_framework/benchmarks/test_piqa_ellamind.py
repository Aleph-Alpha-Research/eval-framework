"""Specification of the German PIQA (EllaMind) tasks.

Each spec test builds the real benchmark (via its ``piqa_ellamind_*_de`` constructor) over a fictional
dataset and asserts the assembled messages, ground truth, and scored completions — so this file reads
as PIQA's prompt spec, with ``composed.py`` an implementation detail. ``test_formatter_hash`` separately
pins the real benchmarks against the real HuggingFace data.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from eval_framework.benchmarks.piqa_ellamind import (
    piqa_ellamind_bpb_de,
    piqa_ellamind_cloze_easy_de,
    piqa_ellamind_cloze_hard_de,
    piqa_ellamind_mc_easy_de,
    piqa_ellamind_mc_hard_de,
)
from eval_framework.contract import Benchmark
from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_names import register_piqa_ellamind_tasks
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

# Registry for this test suite only holding piqa_ellamind tasks
_piqa_ellamind_registry = Registry()
register_piqa_ellamind_tasks(registry=_piqa_ellamind_registry)


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter, NoStripConcatFormatter])
@pytest.mark.parametrize("task_name", _piqa_ellamind_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, registry=_piqa_ellamind_registry)


# ---------------------------------------------------------------------------
# Prompt spec: build the real benchmark over fictional rows, assert the assembled messages
# ---------------------------------------------------------------------------

# Fictional rows in the PIQA format (NOT real dataset examples). Choices are shuffled deterministically
# (seed: goal + correct_solution), which places the correct answer at index 0.
_EVAL_ROW: dict[str, Any] = {
    "goal": "Martin möchte einen Nagel in die Wand schlagen.",
    "correct_solution": "Er verwendet einen Hammer.",
    "easy_distractor": "Er verwendet eine Schere.",
    "hard_distractor": "Er verwendet eine Zange.",
}
_FEWSHOT_ROW: dict[str, Any] = {
    "goal": "Prabhu will Wasser aufkochen.",
    "correct_solution": "Er stellt den Topf auf den Herd.",
    "easy_distractor": "Er schreit den Topf an, bis er warm wird.",
    "hard_distractor": "Er stellt den Topf über Nacht in den Kühlschrank.",
}

_QUESTION = "Ziel: Martin möchte einen Nagel in die Wand schlagen."


@dataclass(frozen=True)
class _ExpectedPrompt:
    messages: list[Message]
    ground_truth: str
    possible_completions: list[str]


# --- Zero-shot: the eval row on its own ---
_MC_EASY = _ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content=f"{_QUESTION}\nA. Er verwendet einen Hammer.\nB. Er verwendet eine Schere.\n"),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    ground_truth=" A",
    possible_completions=[" A", " B"],
)
_MC_HARD = _ExpectedPrompt(
    messages=[
        Message(role=Role.USER, content=f"{_QUESTION}\nA. Er verwendet einen Hammer.\nB. Er verwendet eine Zange.\n"),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    ground_truth=" A",
    possible_completions=[" A", " B"],
)
# Cloze/BPB show no options, so the assembled messages are identical; only the scored completions differ.
_CLOZE_MESSAGES = [
    Message(role=Role.USER, content=f"{_QUESTION}\n"),
    Message(role=Role.ASSISTANT, content="Antwort:"),
]
_CLOZE_EASY = _ExpectedPrompt(
    messages=_CLOZE_MESSAGES,
    ground_truth=" Er verwendet einen Hammer.",
    possible_completions=[" Er verwendet einen Hammer.", " Er verwendet eine Schere."],
)
_CLOZE_HARD = _ExpectedPrompt(
    messages=_CLOZE_MESSAGES,
    ground_truth=" Er verwendet einen Hammer.",
    possible_completions=[" Er verwendet einen Hammer.", " Er verwendet eine Zange."],
)
_BPB = _ExpectedPrompt(
    messages=_CLOZE_MESSAGES,
    ground_truth=" Er verwendet einen Hammer.",
    possible_completions=[" Er verwendet einen Hammer."],  # BPB scores only the gold continuation
)


@pytest.mark.parametrize(
    "make_benchmark, expected",
    [
        pytest.param(piqa_ellamind_mc_easy_de, _MC_EASY, id="mc_easy"),
        pytest.param(piqa_ellamind_mc_hard_de, _MC_HARD, id="mc_hard"),
        pytest.param(piqa_ellamind_cloze_easy_de, _CLOZE_EASY, id="cloze_easy"),
        pytest.param(piqa_ellamind_cloze_hard_de, _CLOZE_HARD, id="cloze_hard"),
        pytest.param(piqa_ellamind_bpb_de, _BPB, id="bpb"),
    ],
)
def test_piqa_zeroshot_prompt(make_benchmark: Callable[..., Benchmark], expected: _ExpectedPrompt) -> None:
    # Given the real PIQA benchmark over a single fictional row
    benchmark = make_benchmark(dataset=DatasetStub({"validation": [_EVAL_ROW]}))
    # When we assemble its first sample (zero-shot)
    sample = first_sample(benchmark, num_fewshot=0)
    # Then the messages, ground truth, and scored completions are exactly:
    assert sample.messages == expected.messages
    assert sample.ground_truth == expected.ground_truth
    assert sample.possible_completions == expected.possible_completions


# --- One-shot: the fewshot row is rendered with its answer, then the eval row (its own zero-shot prompt) ---
_MC_EASY_FEWSHOT_MESSAGES = [
    Message(
        role=Role.USER,
        content="Ziel: Prabhu will Wasser aufkochen.\n"
        "A. Er stellt den Topf auf den Herd.\nB. Er schreit den Topf an, bis er warm wird.\n",
    ),
    Message(role=Role.ASSISTANT, content="Antwort: A"),
]
_MC_HARD_FEWSHOT_MESSAGES = [
    Message(
        role=Role.USER,
        content="Ziel: Prabhu will Wasser aufkochen.\n"
        "A. Er stellt den Topf auf den Herd.\nB. Er stellt den Topf über Nacht in den Kühlschrank.\n",
    ),
    Message(role=Role.ASSISTANT, content="Antwort: A"),
]
_CLOZE_FEWSHOT_MESSAGES = [
    Message(role=Role.USER, content="Ziel: Prabhu will Wasser aufkochen.\n"),
    Message(role=Role.ASSISTANT, content="Antwort: Er stellt den Topf auf den Herd."),
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
        pytest.param(piqa_ellamind_mc_easy_de, _oneshot(_MC_EASY_FEWSHOT_MESSAGES, _MC_EASY), id="mc_easy"),
        pytest.param(piqa_ellamind_mc_hard_de, _oneshot(_MC_HARD_FEWSHOT_MESSAGES, _MC_HARD), id="mc_hard"),
        pytest.param(piqa_ellamind_cloze_easy_de, _oneshot(_CLOZE_FEWSHOT_MESSAGES, _CLOZE_EASY), id="cloze_easy"),
        pytest.param(piqa_ellamind_cloze_hard_de, _oneshot(_CLOZE_FEWSHOT_MESSAGES, _CLOZE_HARD), id="cloze_hard"),
        pytest.param(piqa_ellamind_bpb_de, _oneshot(_CLOZE_FEWSHOT_MESSAGES, _BPB), id="bpb"),
    ],
)
def test_piqa_oneshot_prompt(make_benchmark: Callable[..., Benchmark], expected: _ExpectedPrompt) -> None:
    # Given the real PIQA benchmark over a fewshot row followed by the eval row (fewshot row first so the
    # seed-42 shuffle puts the eval row first, making it the sample and the other the fewshot example)
    benchmark = make_benchmark(dataset=DatasetStub({"validation": [_FEWSHOT_ROW, _EVAL_ROW]}))
    # When we assemble its first sample (one-shot)
    sample = first_sample(benchmark, num_fewshot=1)
    # Then the fewshot example precedes the eval prompt, and ground truth / completions are the eval row's:
    assert sample.messages == expected.messages
    assert sample.ground_truth == expected.ground_truth
    assert sample.possible_completions == expected.possible_completions
