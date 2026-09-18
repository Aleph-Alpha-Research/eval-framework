"""Specification of the composed GPQA tasks.

Each spec test builds the real benchmark over one fictional row and asserts the assembled sample — so this
file reads as GPQA's prompt spec, with ``composed.py`` an implementation detail. GPQA shuffles the correct
answer in among the distractors, seeded from the option texts, so the fictional row below has a fixed,
reproducible option order. ``test_formatter_hash`` pins every variant against the real (gated) HuggingFace
data — it needs Hugging Face auth (the CI hashing job provides a token).
"""

from typing import Any

import pytest

from eval_framework.answer import ExtractedAnswer
from eval_framework.benchmarks.gpqa import (
    _COT_ANSWER_RE,
    _COT_V2_ANSWER_RE,
    GPQA_BENCHMARKS,
    GpqaReader,
    gpqa_diamond_cot,
    gpqa_olmes,
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

# Registry for this test suite only holding the composed gpqa tasks.
_gpqa_registry = Registry()
for _benchmark in GPQA_BENCHMARKS:
    _gpqa_registry.add(_benchmark)


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter, NoStripConcatFormatter])
@pytest.mark.parametrize("task_name", _gpqa_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, registry=_gpqa_registry)


# ---------------------------------------------------------------------------
# Prompt spec: build the real benchmark over one fictional row, assert the assembled sample
# ---------------------------------------------------------------------------

# Fictional GPQA row (NOT a real example). The reader shuffles the four options deterministically to the
# order [4, 6, 5, 3] with the correct answer at position A — pinned in the expectations below.
_EVAL_ROW: dict[str, Any] = {
    "Question": "  What is 2+2?  ",  # surrounding whitespace is stripped by the reader
    "Correct Answer": "4",
    "Incorrect Answer 1": "3",
    "Incorrect Answer 2": "5",
    "Incorrect Answer 3": "6",
}


def test_gpqa_olmes_prompt() -> None:
    # OLMES: a fixed expert-framing preamble, space-prefixed option labels, scored over the letter labels.
    benchmark = gpqa_olmes(dataset=DatasetStub({"train": [_EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0, custom_subjects=["gpqa_extended"])
    assert sample.messages == [
        Message(
            role=Role.USER,
            content="Here are some example questions from experts. An explanation is given before the final "
            "answer. Answer the final question yourself, giving your reasoning beforehand.\n\n"
            "Question: What is 2+2?\n A. 4\n B. 6\n C. 5\n D. 3\n",
        ),
        Message(role=Role.ASSISTANT, content="Answer:"),
    ]
    assert sample.ground_truth == " A"
    assert sample.possible_completions == [" A", " B", " C", " D"]


def test_gpqa_diamond_cot_prompt() -> None:
    # COT: free-form generation — a reasoning frame around the parens-labelled options, no assistant cue.
    benchmark = gpqa_diamond_cot(dataset=DatasetStub({"train": [_EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0, custom_subjects=["gpqa_diamond"])
    assert sample.messages == [
        Message(
            role=Role.USER,
            content="Answer the following multiple-choice question by giving the correct answer letter in "
            "parentheses. Provide CONCISE reasoning for the answer, and make sure to finish the response with "
            '"Therefore, the answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B), (C), (D), (E), etc.'
            "\n\nQuestion: What is 2+2?\n(A) 4\n(B) 6\n(C) 5\n(D) 3"
            "\n\nAnswer the above question and REMEMBER to finish your response with the exact phrase "
            '"Therefore, the answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B), (C), (D), (E), etc.',
        ),
    ]
    assert sample.ground_truth == "A"  # bare letter
    # The completion sample still carries the letters; inert for free-form, kept faithfully.
    assert sample.possible_completions == [" (A)", " (B)", " (C)", " (D)"]


# ---------------------------------------------------------------------------
# Reader: the deterministic, content-seeded option shuffle
# ---------------------------------------------------------------------------


def test_reader_shuffle_is_deterministic() -> None:
    # Given the same item, the shuffle (seeded by the option texts) yields the identical order every time.
    reader = GpqaReader()
    assert reader.read(_EVAL_ROW) == reader.read(dict(_EVAL_ROW))


def test_reader_places_correct_answer_at_correct_index() -> None:
    fields = GpqaReader().read(_EVAL_ROW)
    assert len(fields.choices) == 4
    assert fields.choices[fields.correct_index] == "4"  # the (preprocessed) correct answer


def test_reader_strips_bracketed_markup() -> None:
    # GPQA options carry stray "[...]" markup; the reader removes it before shuffling.
    row = {**_EVAL_ROW, "Correct Answer": "sodium [element] chloride"}
    fields = GpqaReader().read(row)
    assert fields.choices[fields.correct_index] == "sodium chloride"


# ---------------------------------------------------------------------------
# Answer extraction (runs at scoring time, not captured by the formatter hash) — exercised directly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "completion, expected",
    [
        ("Therefore, the answer is (B)", "B"),
        ("Therefore, the answer is (J)", "J"),  # v1 accepts letters up to J
        ("Therefore, the answer is C", "[invalid]"),  # v1 requires parentheses
        ("", "[invalid]"),
    ],
)
def test_gpqa_cot_extracts_parenthesised_letter(completion: str, expected: str) -> None:
    answer = ExtractedAnswer(_COT_ANSWER_RE, ["Question:"])
    assert answer.extract_answer(completion, context=None, ground_truth=None, messages=[]) == expected


@pytest.mark.parametrize(
    "completion, expected",
    [
        ("Therefore, the answer is (B)", "B"),
        ("Therefore, the answer is C", "C"),  # v2 does not require parentheses
        ("the correct answer is (d).", "D"),  # case-insensitive, upper-cased
        ("The answer is (A). Wait, actually the answer is (C).", "C"),  # last match wins
        ("Therefore, the answer is (E)", "[invalid]"),  # v2 only accepts A-D
        ("", "[invalid]"),
    ],
)
def test_gpqa_cot_v2_extracts_leniently(completion: str, expected: str) -> None:
    answer = ExtractedAnswer(_COT_V2_ANSWER_RE, [], last_match=True)
    assert answer.extract_answer(completion, context=None, ground_truth=None, messages=[]) == expected
