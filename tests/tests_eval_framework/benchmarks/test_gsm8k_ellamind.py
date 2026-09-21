"""Specification of the composed German GSM8K (EllaMind) tasks.

Dataset: https://huggingface.co/datasets/ellamind/gsm8k-platinum-multilingual (deu). Few-shot demonstrations
are sampled from the test split and rendered in the German answer format. ``GSM8K_Ellamind_DE_Platinum`` is
free-form (scored on the final integer); ``GSM8K_Ellamind_DE_BPB_Platinum`` scores the single gold solution.
"""

from typing import Any

import pytest

from eval_framework.benchmarks.gsm8k_ellamind import (
    GSM8K_ELLAMIND_BENCHMARKS,
    _extract_final_integer,
    gsm8k_ellamind_de_bpb_platinum,
    gsm8k_ellamind_de_platinum,
)
from eval_framework.contract import Benchmark
from template_formatting.formatter import (
    BaseFormatter,
    ConcatFormatter,
    Llama3Formatter,
    Message,
    NoStripConcatFormatter,
    Role,
)
from tests.tests_eval_framework.benchmarks.utils import DatasetStub, first_sample
from tests.tests_eval_framework.tasks.benchmarks.utils import assert_benchmark_formatter_hash


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
    benchmark = gsm8k_ellamind_de_platinum(dataset=DatasetStub({"test": [_EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0, custom_subjects=["deu"])
    assert sample.messages == [_EVAL_USER, _CUE]
    assert sample.ground_truth == "8"  # the final integer
    assert sample.possible_completions is None  # free-form


def test_generative_oneshot() -> None:
    # The sampled demonstration is rendered in the same German answer format as a full solution.
    benchmark = gsm8k_ellamind_de_platinum(dataset=DatasetStub({"test": [_FEWSHOT_ROW, _EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=1, custom_subjects=["deu"])
    assert sample.messages == [*_DEMO, _EVAL_USER, _CUE]
    assert sample.ground_truth == "8"
    assert sample.possible_completions is None


def test_bpb_zeroshot() -> None:
    # Same assembled prompt as the generative task, but scores the single gold solution.
    benchmark = gsm8k_ellamind_de_bpb_platinum(dataset=DatasetStub({"test": [_EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0, custom_subjects=["deu"])
    assert sample.messages == [_EVAL_USER, _CUE]
    assert sample.ground_truth == _BPB_GOLD
    assert sample.possible_completions == [_BPB_GOLD]


def test_bpb_oneshot() -> None:
    benchmark = gsm8k_ellamind_de_bpb_platinum(dataset=DatasetStub({"test": [_FEWSHOT_ROW, _EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=1, custom_subjects=["deu"])
    assert sample.messages == [*_DEMO, _EVAL_USER, _CUE]
    assert sample.ground_truth == _BPB_GOLD
    assert sample.possible_completions == [_BPB_GOLD]


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
