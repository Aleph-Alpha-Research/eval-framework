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
    GPQA_ELLAMIND_BENCHMARKS,
    gpqa_ellamind_bpb_de,
    gpqa_ellamind_cloze_de,
    gpqa_ellamind_diamond_bpb_de,
    gpqa_ellamind_diamond_cloze_de,
    gpqa_ellamind_diamond_cot_de,
    gpqa_ellamind_diamond_mc_de,
    gpqa_ellamind_mc_de,
    tulu_answer_de,
)
from eval_framework.contract import Benchmark
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter, Message, Role
from tests.tests_eval_framework.benchmarks.utils import DatasetStub, first_sample
from tests.tests_eval_framework.tasks.benchmarks.utils import assert_benchmark_formatter_hash


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter])
@pytest.mark.parametrize("benchmark", GPQA_ELLAMIND_BENCHMARKS, ids=lambda b: b.id())
def test_formatter_hash(benchmark: Benchmark, formatter_cls: type[BaseFormatter]) -> None:
    assert_benchmark_formatter_hash(benchmark, formatter_cls)


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


def test_gpqa_diamond_cot_prompt() -> None:
    # COT: free-form generation, the reasoning instruction is stated twice, no assistant cue,
    # no scored candidates.
    benchmark = gpqa_ellamind_diamond_cot_de(dataset=DatasetStub({"train": [_EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0)
    assert sample.messages == [
        Message(
            role=Role.USER,
            content="Beantworte die folgende Multiple-Choice-Frage, indem du den Buchstaben der richtigen "
            'Antwort in Klammern angibst. Begründe deine Antwort KURZ und beende deine Antwort unbedingt mit "Daher '
            'ist die Antwort (ANTWORTBUCHSTABE)", wobei (ANTWORTBUCHSTABE) einer von (A), (B), (C), (D), (E) usw. '
            "ist.\n\n"
            "Frage: Was ist die SI-Einheit des elektrischen Widerstands?\n"
            "(A) Ohm\n(B) Volt\n(C) Watt\n(D) Ampere\n\n"
            'Beantworte die obige Frage und DENKE DARAN, deine Antwort mit genau dem Satz "Daher ist die Antwort '
            '(ANTWORTBUCHSTABE)" abzuschließen, wobei (ANTWORTBUCHSTABE) einer von (A), (B), (C), (D), (E) usw. ist.',
        ),
    ]
    assert sample.ground_truth == "A"  # bare letter
    assert sample.possible_completions is None  # free-form generation, no candidates


@pytest.mark.parametrize(
    "completion, expected",
    [
        ("Kurze Begründung ... Daher ist die Antwort (C).", "C"),
        # lenient: the parentheses and the German word order are optional, case-insensitive, upper-cased
        ("Die Antwort ist B", "B"),
        ("antwort: d", "D"),
        # anchored on the answer phrase: a bare "Antwort <letter>" in the reasoning is not a conclusion
        ("Daher ist die Antwort (C). Antwort D wäre falsch, weil ...", "C"),
        ("Option A führt zur Antwort B. Daher ist die Antwort (A)", "A"),
        # a model prompted in German may still conclude in English
        ("Therefore, the answer is (C)", "C"),
        # the last conclusion wins, so a restated reminder does not shadow the real answer
        ("Die Antwort ist A ... nein, daher ist die Antwort (D)", "D"),
        # only A-D are accepted, and the reminder's own placeholder is not an answer
        ("Daher ist die Antwort (E)", "[invalid]"),
        ('Beende mit "Daher ist die Antwort (ANTWORTBUCHSTABE)"', "[invalid]"),
        ("Ich bin mir ziemlich sicher.", "[invalid]"),
        ("", "[invalid]"),
    ],
)
def test_gpqa_cot_extracts_the_concluding_letter(completion: str, expected: str) -> None:
    # extract_answer runs at scoring time (not captured by the formatter hash), so exercise it directly.
    answer = tulu_answer_de()
    assert answer.extract_answer(completion, context=None, ground_truth=None, messages=[]) == expected
