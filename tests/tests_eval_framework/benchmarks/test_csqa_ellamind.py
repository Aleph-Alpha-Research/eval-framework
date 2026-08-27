"""Specification of the German CSQA (EllaMind) tasks.

Each spec test builds the real benchmark (via its ``csqa_ellamind_*_de`` constructor) over a fictional
dataset and asserts the assembled messages, ground truth, and scored completions — so this file reads
as CSQA's prompt spec, with ``composed.py`` an implementation detail. ``test_formatter_hash`` separately
pins the real benchmarks against the real HuggingFace data.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from eval_framework.benchmarks.csqa_ellamind import (
    CsqaReader,
    csqa_ellamind_bpb_de,
    csqa_ellamind_cloze_easy_de,
    csqa_ellamind_cloze_hard_de,
    csqa_ellamind_mc_easy_de,
    csqa_ellamind_mc_hard_de,
)
from eval_framework.contract import Benchmark
from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_names import register_csqa_ellamind_tasks
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter, Message, Role
from tests.tests_eval_framework.benchmarks.utils import DatasetStub, first_sample
from tests.tests_eval_framework.tasks.benchmarks.utils import run_formatter_hash_test

# Registry for this test suite only holding csqa_ellamind tasks
_csqa_ellamind_registry = Registry()
register_csqa_ellamind_tasks(registry=_csqa_ellamind_registry)


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter])
@pytest.mark.parametrize("task_name", _csqa_ellamind_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, registry=_csqa_ellamind_registry)


# ---------------------------------------------------------------------------
# Prompt spec: build the real benchmark over fictional rows, assert the assembled messages
# ---------------------------------------------------------------------------

# Fictional rows in the CSQA format (NOT real dataset examples). Choices are shuffled deterministically
# (seed: question + correct_answer).
_EVAL_ROW: dict[str, Any] = {
    "question": "Wo bewahrt man frische Milch am besten auf?",
    "correct_answer": "Im Kühlschrank",
    "easy_distractors": ["Auf dem Mond", "In einem Schuh", "Im Vulkan"],
    "hard_distractors": ["In der Speisekammer", "Auf der Fensterbank", "Im Keller"],
}
_FEWSHOT_ROW: dict[str, Any] = {
    "question": "Womit schreibt man normalerweise auf Papier?",
    "correct_answer": "Mit einem Stift",
    "easy_distractors": ["Mit einer Banane", "Mit einer Wolke", "Mit einem Stein"],
    "hard_distractors": ["Mit einem Pinsel", "Mit Kreide", "Mit einer Tastatur"],
}

_EVAL_Q = "Frage: Wo bewahrt man frische Milch am besten auf?"


@dataclass(frozen=True)
class _ExpectedPrompt:
    messages: list[Message]
    ground_truth: str
    possible_completions: list[str]


# --- Zero-shot ---
_MC_EASY = _ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER, content=f"{_EVAL_Q}\nA. Im Kühlschrank\nB. Auf dem Mond\nC. In einem Schuh\nD. Im Vulkan\n"
        ),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    ground_truth=" A",
    possible_completions=[" A", " B", " C", " D"],
)
_MC_HARD = _ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content=f"{_EVAL_Q}\nA. Im Kühlschrank\nB. In der Speisekammer\nC. Auf der Fensterbank\nD. Im Keller\n",
        ),
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
_CLOZE_EASY = _ExpectedPrompt(
    messages=_CLOZE_MESSAGES,
    ground_truth=" Im Kühlschrank",
    possible_completions=[" Im Kühlschrank", " Auf dem Mond", " In einem Schuh", " Im Vulkan"],
)
_CLOZE_HARD = _ExpectedPrompt(
    messages=_CLOZE_MESSAGES,
    ground_truth=" Im Kühlschrank",
    possible_completions=[" Im Kühlschrank", " In der Speisekammer", " Auf der Fensterbank", " Im Keller"],
)
_BPB = _ExpectedPrompt(
    messages=_CLOZE_MESSAGES,
    ground_truth=" Im Kühlschrank",
    possible_completions=[" Im Kühlschrank"],  # BPB scores only the gold continuation
)


@pytest.mark.parametrize(
    "make_benchmark, expected",
    [
        pytest.param(csqa_ellamind_mc_easy_de, _MC_EASY, id="mc_easy"),
        pytest.param(csqa_ellamind_mc_hard_de, _MC_HARD, id="mc_hard"),
        pytest.param(csqa_ellamind_cloze_easy_de, _CLOZE_EASY, id="cloze_easy"),
        pytest.param(csqa_ellamind_cloze_hard_de, _CLOZE_HARD, id="cloze_hard"),
        pytest.param(csqa_ellamind_bpb_de, _BPB, id="bpb"),
    ],
)
def test_csqa_zeroshot_prompt(make_benchmark: Callable[..., Benchmark], expected: _ExpectedPrompt) -> None:
    benchmark = make_benchmark(dataset=DatasetStub({"validation": [_EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0)
    assert sample.messages == expected.messages
    assert sample.ground_truth == expected.ground_truth
    assert sample.possible_completions == expected.possible_completions


# --- One-shot: fewshot row rendered with its answer, then the eval row's own zero-shot prompt ---
_MC_EASY_FEWSHOT_MESSAGES = [
    Message(
        role=Role.USER,
        content="Frage: Womit schreibt man normalerweise auf Papier?\n"
        "A. Mit einer Wolke\nB. Mit einem Stift\nC. Mit einem Stein\nD. Mit einer Banane\n",
    ),
    Message(role=Role.ASSISTANT, content="Antwort: B"),
]
_MC_HARD_FEWSHOT_MESSAGES = [
    Message(
        role=Role.USER,
        content="Frage: Womit schreibt man normalerweise auf Papier?\n"
        "A. Mit Kreide\nB. Mit einem Stift\nC. Mit einer Tastatur\nD. Mit einem Pinsel\n",
    ),
    Message(role=Role.ASSISTANT, content="Antwort: B"),
]
_CLOZE_FEWSHOT_MESSAGES = [
    Message(role=Role.USER, content="Frage: Womit schreibt man normalerweise auf Papier?\n"),
    Message(role=Role.ASSISTANT, content="Antwort: Mit einem Stift"),
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
        pytest.param(csqa_ellamind_mc_easy_de, _oneshot(_MC_EASY_FEWSHOT_MESSAGES, _MC_EASY), id="mc_easy"),
        pytest.param(csqa_ellamind_mc_hard_de, _oneshot(_MC_HARD_FEWSHOT_MESSAGES, _MC_HARD), id="mc_hard"),
        pytest.param(csqa_ellamind_cloze_easy_de, _oneshot(_CLOZE_FEWSHOT_MESSAGES, _CLOZE_EASY), id="cloze_easy"),
        pytest.param(csqa_ellamind_cloze_hard_de, _oneshot(_CLOZE_FEWSHOT_MESSAGES, _CLOZE_HARD), id="cloze_hard"),
        pytest.param(csqa_ellamind_bpb_de, _oneshot(_CLOZE_FEWSHOT_MESSAGES, _BPB), id="bpb"),
    ],
)
def test_csqa_oneshot_prompt(make_benchmark: Callable[..., Benchmark], expected: _ExpectedPrompt) -> None:
    benchmark = make_benchmark(dataset=DatasetStub({"validation": [_FEWSHOT_ROW, _EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=1)
    assert sample.messages == expected.messages
    assert sample.ground_truth == expected.ground_truth
    assert sample.possible_completions == expected.possible_completions


# ---------------------------------------------------------------------------
# Distractor shuffling: deterministic, correct answer preserved, easy/hard draw from different pools
# ---------------------------------------------------------------------------
def test_csqa_shuffling_is_deterministic_and_uses_expected_distractor_set() -> None:
    item: dict[str, Any] = {
        "question": "Was essen Pandas am liebsten?",
        "correct_answer": "Bambus",
        "easy_distractors": ["Pizza", "Eis", "Schokolade"],
        "hard_distractors": ["Blätter", "Gräser", "Kräuter"],
    }

    easy_1 = CsqaReader("easy").read(item)
    easy_2 = CsqaReader("easy").read(item)
    hard = CsqaReader("hard").read(item)

    # Deterministic for identical input.
    assert (easy_1.choices, easy_1.correct_index) == (easy_2.choices, easy_2.correct_index)
    # Correct index points to the correct answer.
    assert easy_1.choices[easy_1.correct_index] == "Bambus"
    assert hard.choices[hard.correct_index] == "Bambus"
    # Easy and hard draw from different distractor pools.
    assert set(easy_1.choices) == {"Bambus", "Pizza", "Eis", "Schokolade"}
    assert set(hard.choices) == {"Bambus", "Blätter", "Gräser", "Kräuter"}
