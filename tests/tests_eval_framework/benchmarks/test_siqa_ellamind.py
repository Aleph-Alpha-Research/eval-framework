"""Specification of the German Social IQa (EllaMind) tasks.

Each spec test builds the real benchmark (via its ``siqa_ellamind_*_de`` constructor) over a fictional
dataset and asserts the assembled messages, ground truth, and scored completions — so this file reads
as SIQA's prompt spec, with ``composed.py`` an implementation detail. ``test_formatter_hash`` separately
pins the real benchmarks against the real HuggingFace data. The shown question is the context followed
by the question.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from eval_framework.benchmarks.siqa_ellamind import (
    siqa_ellamind_bpb_de,
    siqa_ellamind_cloze_easy_de,
    siqa_ellamind_cloze_hard_de,
    siqa_ellamind_mc_easy_de,
    siqa_ellamind_mc_hard_de,
)
from eval_framework.contract import Benchmark
from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_names import register_siqa_ellamind_tasks
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter, Message, Role
from tests.tests_eval_framework.benchmarks.utils import DatasetStub, first_sample
from tests.tests_eval_framework.tasks.benchmarks.utils import run_formatter_hash_test

# Registry for this test suite only holding siqa_ellamind tasks
_siqa_ellamind_registry = Registry()
register_siqa_ellamind_tasks(registry=_siqa_ellamind_registry)


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter])
@pytest.mark.parametrize("task_name", _siqa_ellamind_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, registry=_siqa_ellamind_registry)


# ---------------------------------------------------------------------------
# Prompt spec: build the real benchmark over fictional rows, assert the assembled messages
# ---------------------------------------------------------------------------

# Fictional rows in the SIQA format (NOT real dataset examples). Choices are shuffled deterministically
# (seed: question + correct_answer).
_EVAL_ROW: dict[str, Any] = {
    "context": "Max kommt nach Hause und findet sein Zimmer aufgeräumt.",
    "question": "Was hat Max als nächstes getan?",
    "correct_answer": "Er hat sich bedankt.",
    "easy_distractors": ["Er ist wütend geworden.", "Er ist gegangen."],
    "hard_distractors": ["Er hat gegessen.", "Er hat geschlafen."],
}
_FEWSHOT_ROW: dict[str, Any] = {
    "context": "Lisa hat ihr Buch vergessen.",
    "question": "Wie hat Lisa sich gefühlt?",
    "correct_answer": "Ärgerlich.",
    "easy_distractors": ["Glücklich.", "Müde."],
    "hard_distractors": ["Neugierig.", "Aufgeregt."],
}

# The shown question is "Frage: " + context + " " + question.
_EVAL_Q = "Frage: Max kommt nach Hause und findet sein Zimmer aufgeräumt. Was hat Max als nächstes getan?"
_FEWSHOT_Q = "Frage: Lisa hat ihr Buch vergessen. Wie hat Lisa sich gefühlt?"


@dataclass(frozen=True)
class _ExpectedPrompt:
    messages: list[Message]
    ground_truth: str
    possible_completions: list[str]


# --- Zero-shot ---
_MC_EASY = _ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content=f"{_EVAL_Q}\nA. Er ist wütend geworden.\nB. Er hat sich bedankt.\nC. Er ist gegangen.\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    ground_truth=" B",
    possible_completions=[" A", " B", " C"],
)
_MC_HARD = _ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER, content=f"{_EVAL_Q}\nA. Er hat gegessen.\nB. Er hat sich bedankt.\nC. Er hat geschlafen.\n"
        ),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    ground_truth=" B",
    possible_completions=[" A", " B", " C"],
)
# Cloze/BPB show no options, so the assembled messages are identical; only the scored completions differ.
_CLOZE_MESSAGES = [
    Message(role=Role.USER, content=f"{_EVAL_Q}\n"),
    Message(role=Role.ASSISTANT, content="Antwort:"),
]
_CLOZE_EASY = _ExpectedPrompt(
    messages=_CLOZE_MESSAGES,
    ground_truth=" Er hat sich bedankt.",
    possible_completions=[" Er ist wütend geworden.", " Er hat sich bedankt.", " Er ist gegangen."],
)
_CLOZE_HARD = _ExpectedPrompt(
    messages=_CLOZE_MESSAGES,
    ground_truth=" Er hat sich bedankt.",
    possible_completions=[" Er hat gegessen.", " Er hat sich bedankt.", " Er hat geschlafen."],
)
_BPB = _ExpectedPrompt(
    messages=_CLOZE_MESSAGES,
    ground_truth=" Er hat sich bedankt.",
    possible_completions=[" Er hat sich bedankt."],  # BPB scores only the gold continuation
)


@pytest.mark.parametrize(
    "make_benchmark, expected",
    [
        pytest.param(siqa_ellamind_mc_easy_de, _MC_EASY, id="mc_easy"),
        pytest.param(siqa_ellamind_mc_hard_de, _MC_HARD, id="mc_hard"),
        pytest.param(siqa_ellamind_cloze_easy_de, _CLOZE_EASY, id="cloze_easy"),
        pytest.param(siqa_ellamind_cloze_hard_de, _CLOZE_HARD, id="cloze_hard"),
        pytest.param(siqa_ellamind_bpb_de, _BPB, id="bpb"),
    ],
)
def test_siqa_zeroshot_prompt(make_benchmark: Callable[..., Benchmark], expected: _ExpectedPrompt) -> None:
    benchmark = make_benchmark(dataset=DatasetStub({"validation": [_EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0)
    assert sample.messages == expected.messages
    assert sample.ground_truth == expected.ground_truth
    assert sample.possible_completions == expected.possible_completions


# --- One-shot: fewshot row rendered with its answer, then the eval row's own zero-shot prompt ---
_MC_EASY_FEWSHOT_MESSAGES = [
    Message(role=Role.USER, content=f"{_FEWSHOT_Q}\nA. Ärgerlich.\nB. Müde.\nC. Glücklich.\n"),
    Message(role=Role.ASSISTANT, content="Antwort: A"),
]
_MC_HARD_FEWSHOT_MESSAGES = [
    Message(role=Role.USER, content=f"{_FEWSHOT_Q}\nA. Ärgerlich.\nB. Aufgeregt.\nC. Neugierig.\n"),
    Message(role=Role.ASSISTANT, content="Antwort: A"),
]
_CLOZE_FEWSHOT_MESSAGES = [
    Message(role=Role.USER, content=f"{_FEWSHOT_Q}\n"),
    Message(role=Role.ASSISTANT, content="Antwort: Ärgerlich."),
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
        pytest.param(siqa_ellamind_mc_easy_de, _oneshot(_MC_EASY_FEWSHOT_MESSAGES, _MC_EASY), id="mc_easy"),
        pytest.param(siqa_ellamind_mc_hard_de, _oneshot(_MC_HARD_FEWSHOT_MESSAGES, _MC_HARD), id="mc_hard"),
        pytest.param(siqa_ellamind_cloze_easy_de, _oneshot(_CLOZE_FEWSHOT_MESSAGES, _CLOZE_EASY), id="cloze_easy"),
        pytest.param(siqa_ellamind_cloze_hard_de, _oneshot(_CLOZE_FEWSHOT_MESSAGES, _CLOZE_HARD), id="cloze_hard"),
        pytest.param(siqa_ellamind_bpb_de, _oneshot(_CLOZE_FEWSHOT_MESSAGES, _BPB), id="bpb"),
    ],
)
def test_siqa_oneshot_prompt(make_benchmark: Callable[..., Benchmark], expected: _ExpectedPrompt) -> None:
    benchmark = make_benchmark(dataset=DatasetStub({"validation": [_FEWSHOT_ROW, _EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=1)
    assert sample.messages == expected.messages
    assert sample.ground_truth == expected.ground_truth
    assert sample.possible_completions == expected.possible_completions
