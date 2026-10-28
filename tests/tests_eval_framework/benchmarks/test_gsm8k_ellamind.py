"""Specification of the composed multilingual GSM8K Platinum (EllaMind) tasks.

Dataset: https://huggingface.co/datasets/ellamind/gsm8k-platinum-multilingual, one config per language. Few-shot
demonstrations are sampled from the test split and rendered in the language's answer format.
``GSM8K_Ellamind_{LANG}_Platinum`` is free-form (scored on the final integer);
``GSM8K_Ellamind_{LANG}_BPB_Platinum`` scores the single gold solution. The German tasks are the fully spelled
out specification; the per-language tests check that every language gets its own prefix, cue and
final-answer line around the same scaffolding.
"""

from typing import Any

import pytest

from eval_framework.benchmarks.gsm8k_ellamind import (
    _FINAL_ANSWER_LINE,
    _LANGUAGES,
    GSM8K_ELLAMIND_BENCHMARKS,
    LanguageCode,
    _extract_final_integer,
    gsm8k_ellamind_bpb_platinum,
    gsm8k_ellamind_platinum,
)
from eval_framework.benchmarks.utils.ellamind_languages import ellamind_language
from eval_framework.contract import Benchmark
from eval_framework.tasks.task_style import question_cue_text
from template_formatting.formatter import (
    BaseFormatter,
    ConcatFormatter,
    Llama3Formatter,
    Message,
    NoStripConcatFormatter,
    Role,
)
from tests.tests_eval_framework.benchmarks.utils import DatasetStub, assert_benchmark_formatter_hash, first_sample


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter, NoStripConcatFormatter])
@pytest.mark.parametrize("benchmark", GSM8K_ELLAMIND_BENCHMARKS, ids=lambda b: b.id())
def test_formatter_hash(benchmark: Benchmark, formatter_cls: type[BaseFormatter]) -> None:
    assert_benchmark_formatter_hash(benchmark, formatter_cls)


# ---------------------------------------------------------------------------
# Prompt spec (fictional German rows; NOT real examples)
# ---------------------------------------------------------------------------

_EVAL_ROW: dict[str, Any] = {
    "question": "Du hast 10 Äpfel und gibst 2 weg. Wie viele hast du noch?",
    "solution": "Wenn ich 10 Äpfel habe und 2 weggebe, habe ich 10 - 2 = 8 Äpfel übrig.",
    "final_answer": "8",
}
_FEWSHOT_ROW: dict[str, Any] = {
    "question": "Du hast 5 Orangen und gibst 3 weg. Wie viele hast du noch?",
    "solution": "Wenn ich 5 Orangen habe und 3 weggebe, habe ich 5 - 3 = 2 Orangen übrig.",
    "final_answer": "2",
}

_EVAL_USER = Message(role=Role.USER, content="Frage: Du hast 10 Äpfel und gibst 2 weg. Wie viele hast du noch?\n")
_CUE = Message(role=Role.ASSISTANT, content="Antwort:")
_DEMO = [
    Message(role=Role.USER, content="Frage: Du hast 5 Orangen und gibst 3 weg. Wie viele hast du noch?\n"),
    Message(
        role=Role.ASSISTANT,
        content="Antwort: Wenn ich 5 Orangen habe und 3 weggebe, habe ich 5 - 3 = 2 Orangen übrig. "
        "Daher ist die Antwort 2.",
    ),
]
# The BPB scored candidate is the eval item's own gold solution line (raw final answer).
_BPB_GOLD = " Wenn ich 10 Äpfel habe und 2 weggebe, habe ich 10 - 2 = 8 Äpfel übrig. Daher ist die Antwort 8."


def test_generative_zeroshot() -> None:
    benchmark = gsm8k_ellamind_platinum("DE", dataset=DatasetStub({"test": [_EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0, custom_subjects=["deu"])
    assert sample.messages == [_EVAL_USER, _CUE]
    assert sample.ground_truth == "8"  # the final integer
    assert sample.possible_completions is None  # free-form


def test_generative_oneshot() -> None:
    # The sampled demonstration is rendered in the same German answer format as a full solution.
    benchmark = gsm8k_ellamind_platinum("DE", dataset=DatasetStub({"test": [_FEWSHOT_ROW, _EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=1, custom_subjects=["deu"])
    assert sample.messages == [*_DEMO, _EVAL_USER, _CUE]
    assert sample.ground_truth == "8"
    assert sample.possible_completions is None


def test_bpb_zeroshot() -> None:
    # Same assembled prompt as the generative task, but scores the single gold solution.
    benchmark = gsm8k_ellamind_bpb_platinum("DE", dataset=DatasetStub({"test": [_EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0, custom_subjects=["deu"])
    assert sample.messages == [_EVAL_USER, _CUE]
    assert sample.ground_truth == _BPB_GOLD
    assert sample.possible_completions == [_BPB_GOLD]


def test_bpb_oneshot() -> None:
    benchmark = gsm8k_ellamind_bpb_platinum("DE", dataset=DatasetStub({"test": [_FEWSHOT_ROW, _EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=1, custom_subjects=["deu"])
    assert sample.messages == [*_DEMO, _EVAL_USER, _CUE]
    assert sample.ground_truth == _BPB_GOLD
    assert sample.possible_completions == [_BPB_GOLD]


# ---------------------------------------------------------------------------
# Every language: its own prefix, cue and final-answer line around the same scaffolding
# ---------------------------------------------------------------------------

# Language-agnostic fictional rows: the per-language tests only check the scaffolding around them. The eval
# row's final answer carries a thousands separator (dropped from the ground truth, kept in the BPB target).
_LANG_EVAL_ROW: dict[str, Any] = {"question": "Q?", "solution": "S.", "final_answer": "1,000"}
_LANG_DEMO_ROW: dict[str, Any] = {"question": "D?", "solution": "T.", "final_answer": "2"}


def _expected_messages(language: LanguageCode) -> list[Message]:
    prefix, cue = question_cue_text(ellamind_language(language).language)
    demo_answer = f"T. {_FINAL_ANSWER_LINE[language].format(answer='2')}"
    return [
        Message(role=Role.USER, content=f"{prefix}D?\n"),
        Message(role=Role.ASSISTANT, content=f"{cue} {demo_answer}"),
        Message(role=Role.USER, content=f"{prefix}Q?\n"),
        Message(role=Role.ASSISTANT, content=cue),
    ]


@pytest.mark.parametrize("language", _LANGUAGES)
def test_generative_per_language(language: LanguageCode) -> None:
    config = ellamind_language(language).config
    benchmark = gsm8k_ellamind_platinum(language, dataset=DatasetStub({"test": [_LANG_DEMO_ROW, _LANG_EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=1, custom_subjects=[config])
    assert benchmark.id() == f"GSM8K_Ellamind_{language}_Platinum"
    assert sample.messages == _expected_messages(language)
    assert sample.ground_truth == "1000"
    assert sample.possible_completions is None


@pytest.mark.parametrize("language", _LANGUAGES)
def test_bpb_per_language(language: LanguageCode) -> None:
    config = ellamind_language(language).config
    benchmark = gsm8k_ellamind_bpb_platinum(language, dataset=DatasetStub({"test": [_LANG_DEMO_ROW, _LANG_EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=1, custom_subjects=[config])
    gold = f" S. {_FINAL_ANSWER_LINE[language].format(answer='1,000')}"
    assert benchmark.id() == f"GSM8K_Ellamind_{language}_BPB_Platinum"
    assert sample.messages == _expected_messages(language)
    assert sample.ground_truth == gold
    assert sample.possible_completions == [gold]


def test_every_language_has_a_final_answer_line() -> None:
    assert set(_FINAL_ANSWER_LINE) == set(_LANGUAGES)
    assert all("{answer}" in line for line in _FINAL_ANSWER_LINE.values())


@pytest.mark.parametrize(
    "completion, expected",
    [
        ("8", "8"),
        ("-8", "-8"),
        ("+8", "+8"),
        ("Die Antwort ist 8.", "8"),
        ("1000", "1000"),
        ("1,000", "1000"),  # thousands separators dropped
        ("1.000", "1000"),
        ("Die Antwort ist 1,200.", "1200"),
        ("Zuerst dachte ich die Antwort ist 0, aber dann habe ich gemerkt, dass die Antwort -8 ist.", "-8"),
        ("Leider weiß ich die Antwort nicht.", "[invalid]"),
        ("", "[invalid]"),
    ],
)
def test_extracts_final_integer(completion: str, expected: str) -> None:
    assert _extract_final_integer(completion) == expected
