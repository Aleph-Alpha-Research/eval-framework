"""Specification of the composed MMLU-Pro tasks.

Each spec test builds the real benchmark over one fictional row and asserts the assembled sample — so this
file reads as MMLU-Pro's prompt spec, with ``composed.py`` an implementation detail. The subject label comes
from the evaluated subject (a category); the reader takes the options as-is. ``test_formatter_hash`` pins the
variants against the real HuggingFace data. The loglikelihood variants deliberately score a fixed ten letters
A–J even when a question lists fewer options (a preserved MMLU-Pro quirk); the spec below shows that.
"""

from typing import Any

import pytest

from eval_framework.answer import ExtractedAnswer
from eval_framework.benchmarks.mmlu_pro import (
    _COT_V2_ANSWER_RE,
    MMLU_PRO_BENCHMARKS,
    mmlu_pro,
    mmlu_pro_cot,
    mmlu_pro_idk,
    mmlu_pro_olmes,
)
from eval_framework.tasks.registry import Registry
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

# Registry for this test suite only holding the composed mmlu_pro tasks.
_mmlu_pro_registry = Registry()
for _benchmark in MMLU_PRO_BENCHMARKS:
    _mmlu_pro_registry.add(_benchmark)


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter, NoStripConcatFormatter])
@pytest.mark.parametrize("task_name", _mmlu_pro_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, registry=_mmlu_pro_registry)


# ---------------------------------------------------------------------------
# Prompt spec: build the real benchmark over one fictional row, assert the assembled sample
# ---------------------------------------------------------------------------

# Fictional MMLU-Pro row (NOT a real example). Only four options, to show the fixed-ten-letter scoring.
_EVAL_ROW: dict[str, Any] = {
    "question": "  What is 2 + 2?  ",  # surrounding whitespace is stripped by the reader
    "options": ["3", "4", "5", "6"],
    "answer_index": 1,
}
_SUBJECT = "business"
_TEN_LETTERS = [" A", " B", " C", " D", " E", " F", " G", " H", " I", " J"]


def test_mmlu_pro_prompt() -> None:
    # Base: letter labels, subject preamble; scored over a fixed ten letters despite only four options.
    benchmark = mmlu_pro(dataset=DatasetStub({"test": [_EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0, custom_subjects=[_SUBJECT])
    assert sample.messages == [
        Message(
            role=Role.USER,
            content="The following are multiple choice questions (with answers) about business.\n\n"
            "What is 2 + 2?\nA. 3\nB. 4\nC. 5\nD. 6\n",
        ),
        Message(role=Role.ASSISTANT, content="Answer:"),
    ]
    assert sample.ground_truth == " B"
    assert sample.possible_completions == _TEN_LETTERS  # ten candidates for four options — a preserved quirk


def test_mmlu_pro_olmes_prompt() -> None:
    # OLMES: option labels are space-prefixed (" A." not "A.").
    benchmark = mmlu_pro_olmes(dataset=DatasetStub({"test": [_EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0, custom_subjects=[_SUBJECT])
    assert sample.messages == [
        Message(
            role=Role.USER,
            content="The following are multiple choice questions (with answers) about business.\n\n"
            "What is 2 + 2?\n A. 3\n B. 4\n C. 5\n D. 6\n",
        ),
        Message(role=Role.ASSISTANT, content="Answer:"),
    ]
    assert sample.ground_truth == " B"
    assert sample.possible_completions == _TEN_LETTERS


def test_mmlu_pro_idk_prompt() -> None:
    # IDK: an abstention option "?" is scored alongside the fixed ten letters; the preamble adds the guidance.
    benchmark = mmlu_pro_idk(dataset=DatasetStub({"test": [_EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0, custom_subjects=[_SUBJECT])
    assert sample.messages == [
        Message(
            role=Role.USER,
            content="The following are multiple choice questions (with answers) about business. "
            "Answer only if you are confident, since mistakes may be penalised, while correct answers receive "
            "points. It is acceptable to answer with '?' if you are unsure, and you will receive 0 points.\n\n"
            "What is 2 + 2?\nA. 3\nB. 4\nC. 5\nD. 6\n",
        ),
        Message(role=Role.ASSISTANT, content="Answer:"),
    ]
    assert sample.ground_truth == " B"
    assert sample.possible_completions == [*_TEN_LETTERS, " ?"]


def test_mmlu_pro_cot_prompt() -> None:
    # COT: free-form generation — a reasoning frame around the question, no assistant cue.
    benchmark = mmlu_pro_cot(dataset=DatasetStub({"test": [_EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0, custom_subjects=[_SUBJECT])
    assert sample.messages == [
        Message(
            role=Role.USER,
            content="The following are multiple choice questions (with answers) about business.\n\n"
            "Answer the following multiple-choice question by giving the correct answer letter in parentheses. "
            "Provide CONCISE reasoning for the answer, and make sure to finish the response with "
            '"Therefore, the answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B), (C), (D), (E), etc.'
            "\n\nQuestion: What is 2 + 2?\n(A) 3\n(B) 4\n(C) 5\n(D) 6"
            "\n\nAnswer the above question and REMEMBER to finish your response with the exact phrase "
            '"Therefore, the answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B), (C), (D), (E), etc.',
        ),
    ]
    assert sample.ground_truth == "B"  # bare letter
    # The completion sample still carries the base's ten scored letters — inert for free-form, kept faithfully.
    assert sample.possible_completions == _TEN_LETTERS


@pytest.mark.parametrize(
    "completion, expected",
    [
        ("Therefore, the answer is (J)", "J"),
        ("Therefore, the answer is J", "J"),
        ("**Therefore, the answer is (h).**", "H"),  # case-insensitive, upper-cased
        ("The answer is (A). Wait, actually the answer is (C).", "C"),  # last match wins
        ("Question: what is 2+2?", "[invalid]"),
        ("", "[invalid]"),
    ],
)
def test_mmlu_pro_cot_v2_extracts_leniently(completion: str, expected: str) -> None:
    # extract_answer runs at scoring time (not captured by the formatter hash), so exercise it directly.
    answer = ExtractedAnswer(_COT_V2_ANSWER_RE, [], last_match=True)
    assert answer.extract_answer(completion, context=None, ground_truth=None, messages=[]) == expected
