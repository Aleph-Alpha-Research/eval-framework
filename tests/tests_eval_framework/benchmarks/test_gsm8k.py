"""Specification of the composed GSM8K tasks.

Both variants use eight fixed exemplars (no dataset sampling) and OLMES answer normalisation.
``GSM8K_OLMES`` generates a solution and is scored on its final number; ``GSM8KBPB`` scores the
bits-per-byte of the single normalised gold solution. ``test_formatter_hash`` pins both against the real
HuggingFace data.
"""

from typing import Any

import pytest

from eval_framework.benchmarks.gsm8k import (
    FEWSHOT_ITEMS,
    GSM8K_BENCHMARKS,
    _clean_short_answer,
    _normalize_answer_str,
    gsm8k_bpb,
    gsm8k_olmes,
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
@pytest.mark.parametrize("benchmark", GSM8K_BENCHMARKS, ids=lambda b: b.id())
def test_formatter_hash(benchmark: Benchmark, formatter_cls: type[BaseFormatter]) -> None:
    assert_benchmark_formatter_hash(benchmark, formatter_cls)


# ---------------------------------------------------------------------------
# Prompt spec: both variants pin the shot count to the eight predefined exemplars
# ---------------------------------------------------------------------------

_EVAL_ROW: dict[str, Any] = {
    "question": "If there are 2 apples and 3 oranges, how many fruits are there?",
    "answer": "There are 2 apples and 3 oranges. So there are 2 + 3 = 5 fruits. #### 5",
}


def test_gsm8k_olmes_prompt() -> None:
    # Free-form: eight solved exemplars then the eval question, ending on "Answer:" for the model to continue.
    benchmark = gsm8k_olmes(dataset=DatasetStub({"test": [_EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0, custom_subjects=["main"])  # count is pinned to 8
    assert len(sample.messages) == 2 * len(FEWSHOT_ITEMS) + 1
    assert sample.messages[0].role == Role.USER
    assert sample.messages[0].content == f"Question: {FEWSHOT_ITEMS[0]['question']}\nAnswer:"
    assert sample.messages[1].content == _normalize_answer_str(FEWSHOT_ITEMS[0])
    assert sample.messages[-1] == Message(role=Role.USER, content=f"Question: {_EVAL_ROW['question']}\nAnswer:")
    assert sample.ground_truth == "5"  # the final number
    assert sample.possible_completions is None  # free-form generation


def test_gsm8k_bpb_prompt() -> None:
    # BPB: the "Answer:" cue is a separate assistant turn; the single scored candidate is the gold solution.
    benchmark = gsm8k_bpb(dataset=DatasetStub({"test": [_EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0, custom_subjects=["main"])
    assert len(sample.messages) == 2 * len(FEWSHOT_ITEMS) + 2
    assert sample.messages[0].content == f"Question: {FEWSHOT_ITEMS[0]['question']}\n"
    assert sample.messages[1].content == f"Answer:{_normalize_answer_str(FEWSHOT_ITEMS[0])}"
    assert sample.messages[-2].role == Role.USER
    assert sample.messages[-2].content == f"Question: {_EVAL_ROW['question']}\n"
    assert sample.messages[-1].role == Role.ASSISTANT
    assert sample.messages[-1].content == "Answer:"
    normalized = _normalize_answer_str(_EVAL_ROW)
    assert sample.ground_truth == normalized
    assert sample.possible_completions == [normalized]  # BPB scores the gold solution only


# ---------------------------------------------------------------------------
# OLMES answer helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("So there must have been 21 - 15 = 6.\n#### 6", "6"),
        ("The total is 1,234 dollars", "1234"),  # commas stripped
        ("no numbers here", "no numbers here"),  # falls back to the raw text
    ],
)
def test_clean_short_answer(text: str, expected: str) -> None:
    assert _clean_short_answer(text) == expected


def test_normalize_answer_str_reformats_into_a_sentence() -> None:
    item = {"answer": "There are 2 apples and 3 oranges. So there are 2 + 3 = 5 fruits. #### 5"}
    assert _normalize_answer_str(item) == (
        " There are 2 apples and 3 oranges. So there are 2 + 3 = 5 fruits. So the answer is 5."
    )
