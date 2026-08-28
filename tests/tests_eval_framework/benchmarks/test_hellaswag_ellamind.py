"""Specification of the German HellaSwag (EllaMind) tasks.

Each spec test builds the real benchmark (via its ``hellaswag_ellamind_*_de`` constructor) over a fictional
dataset and asserts the assembled messages, ground truth, and scored completions — so this file reads as
HellaSwag's prompt spec, with ``composed.py`` an implementation detail. The rows are fictional so this open
source codebase does not leak the real dataset into training data. HellaSwag is sentence-completion: no
question prefix and no assistant cue, so the prompt is just the partial sentence and the model scores full
endings. ``test_formatter_hash`` separately pins the real benchmarks against the real HuggingFace data.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from eval_framework.benchmarks.hellaswag_ellamind import (
    hellaswag_ellamind_bpb_de,
    hellaswag_ellamind_easy_de,
    hellaswag_ellamind_hard_de,
)
from eval_framework.contract import Benchmark
from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_names import register_hellaswag_ellamind_tasks
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

# Registry for this test suite only holding hellaswag_ellamind tasks
_hellaswag_ellamind_registry = Registry()
register_hellaswag_ellamind_tasks(registry=_hellaswag_ellamind_registry)


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter, NoStripConcatFormatter])
@pytest.mark.parametrize("task_name", _hellaswag_ellamind_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, registry=_hellaswag_ellamind_registry)


# ---------------------------------------------------------------------------
# Prompt spec: build the real benchmark over fictional rows, assert the assembled messages
# ---------------------------------------------------------------------------

# Fictional rows in the HellaSwag format (NOT real dataset examples). Endings are shuffled deterministically
# (seed: context + correct_ending).
_EVAL_ROW: dict[str, Any] = {
    "activity": "Kochen",
    "context": "Die Zwiebeln werden in der Pfanne angebraten",
    "correct_ending": "bis sie goldbraun sind.",
    "easy_distractors": ["mit einem Schraubenzieher.", "auf dem Dach.", "im Schwimmbad."],
    "hard_distractors": ["bis sie gefroren sind.", "bis sie roh sind.", "bis sie trocken sind."],
}
_FEWSHOT_ROW: dict[str, Any] = {
    "activity": "Sport",
    "context": "Der Spieler rennt über das Feld",
    "correct_ending": "und macht ein Tor.",
    "easy_distractors": ["und liest ein Buch.", "und kocht Suppe.", "und schläft ein."],
    "hard_distractors": ["und springt ins Wasser.", "und setzt sich hin.", "und kommt nicht an."],
}


@dataclass(frozen=True)
class _ExpectedPrompt:
    messages: list[Message]
    ground_truth: str
    possible_completions: list[str]


# Sentence-completion: the prompt is just the partial sentence, with no assistant cue after it.
_ZEROSHOT_MESSAGES = [Message(role=Role.USER, content="Kochen: Die Zwiebeln werden in der Pfanne angebraten")]

# Easy/hard/BPB share the same prompt and ground truth; only the scored completions differ.
_EASY = _ExpectedPrompt(
    messages=_ZEROSHOT_MESSAGES,
    ground_truth=" bis sie goldbraun sind.",
    possible_completions=[
        " auf dem Dach.",
        " bis sie goldbraun sind.",
        " mit einem Schraubenzieher.",
        " im Schwimmbad.",
    ],
)
_HARD = _ExpectedPrompt(
    messages=_ZEROSHOT_MESSAGES,
    ground_truth=" bis sie goldbraun sind.",
    possible_completions=[
        " bis sie roh sind.",
        " bis sie goldbraun sind.",
        " bis sie gefroren sind.",
        " bis sie trocken sind.",
    ],
)
_BPB = _ExpectedPrompt(
    messages=_ZEROSHOT_MESSAGES,
    ground_truth=" bis sie goldbraun sind.",
    possible_completions=[" bis sie goldbraun sind."],  # BPB scores only the gold continuation
)


@pytest.mark.parametrize(
    "make_benchmark, expected",
    [
        pytest.param(hellaswag_ellamind_easy_de, _EASY, id="easy"),
        pytest.param(hellaswag_ellamind_hard_de, _HARD, id="hard"),
        pytest.param(hellaswag_ellamind_bpb_de, _BPB, id="bpb"),
    ],
)
def test_hellaswag_zeroshot_prompt(make_benchmark: Callable[..., Benchmark], expected: _ExpectedPrompt) -> None:
    benchmark = make_benchmark(dataset=DatasetStub({"validation": [_EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0)
    assert sample.messages == expected.messages
    assert sample.ground_truth == expected.ground_truth
    assert sample.possible_completions == expected.possible_completions


# One-shot: the fewshot partial sentence, its full ending as the assistant turn, then the eval prompt.
_FEWSHOT_MESSAGES = [
    Message(role=Role.USER, content="Sport: Der Spieler rennt über das Feld"),
    Message(role=Role.ASSISTANT, content=" und macht ein Tor."),
]


def _oneshot(eval_expected: _ExpectedPrompt) -> _ExpectedPrompt:
    return _ExpectedPrompt(
        messages=[*_FEWSHOT_MESSAGES, *eval_expected.messages],
        ground_truth=eval_expected.ground_truth,
        possible_completions=eval_expected.possible_completions,
    )


@pytest.mark.parametrize(
    "make_benchmark, expected",
    [
        pytest.param(hellaswag_ellamind_easy_de, _oneshot(_EASY), id="easy"),
        pytest.param(hellaswag_ellamind_hard_de, _oneshot(_HARD), id="hard"),
        pytest.param(hellaswag_ellamind_bpb_de, _oneshot(_BPB), id="bpb"),
    ],
)
def test_hellaswag_oneshot_prompt(make_benchmark: Callable[..., Benchmark], expected: _ExpectedPrompt) -> None:
    benchmark = make_benchmark(dataset=DatasetStub({"validation": [_FEWSHOT_ROW, _EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=1)
    assert sample.messages == expected.messages
    assert sample.ground_truth == expected.ground_truth
    assert sample.possible_completions == expected.possible_completions
