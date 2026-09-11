"""Specification of the composed MMLU tasks.

Each spec test builds the real benchmark over a fictional dataset and asserts the assembled sample — so
this file reads as MMLU's prompt spec, with ``composed.py`` an implementation detail. The subject label
is an underscored config name; the preamble reads it as prose (except ``MMLU_IDK``, which keeps the raw
key). ``test_formatter_hash`` separately pins the composed variants against the real HuggingFace data.
``MMLU_COT`` is not here — it is still a BaseTask (generative), covered by the legacy ``test_mmlu``.
"""

from collections.abc import Callable
from typing import Any

import pytest

from eval_framework.benchmarks.mmlu import MMLU_BENCHMARKS, mmlu, mmlu_full_text, mmlu_idk, mmlu_olmes
from eval_framework.contract import Benchmark
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

# Registry for this test suite only holding the composed mmlu tasks.
_mmlu_registry = Registry()
for _benchmark in MMLU_BENCHMARKS:
    _mmlu_registry.add(_benchmark)


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter, NoStripConcatFormatter])
@pytest.mark.parametrize("task_name", _mmlu_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, registry=_mmlu_registry)


# ---------------------------------------------------------------------------
# Prompt spec: build the real benchmark over one fictional row, assert the assembled sample
# ---------------------------------------------------------------------------

# Fictional row in the MMLU format (NOT a real dataset example). The subject is the HF config name.
_EVAL_ROW: dict[str, Any] = {
    "subject": "abstract_algebra",
    "question": "  What is 2 + 2?  ",  # surrounding whitespace is stripped by the reader
    "choices": ["3", "4", "5", "6"],
    "answer": 1,
}
_SUBJECT = "abstract_algebra"


def test_mmlu_prompt() -> None:
    # Base MMLU: letter labels, humanized subject preamble.
    benchmark = mmlu(dataset=DatasetStub({"test": [_EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0)
    assert sample.messages == [
        Message(
            role=Role.USER,
            content="The following are multiple choice questions (with answers) about abstract algebra.\n\n"
            "Question: What is 2 + 2?\nA. 3\nB. 4\nC. 5\nD. 6\n",
        ),
        Message(role=Role.ASSISTANT, content="Answer:"),
    ]
    assert sample.ground_truth == " B"
    assert sample.possible_completions == [" A", " B", " C", " D"]


def test_mmlu_olmes_prompt() -> None:
    # OLMES: option labels are space-prefixed (" A." not "A.").
    benchmark = mmlu_olmes(dataset=DatasetStub({"test": [_EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0)
    assert sample.messages == [
        Message(
            role=Role.USER,
            content="The following are multiple choice questions (with answers) about abstract algebra.\n\n"
            "Question: What is 2 + 2?\n A. 3\n B. 4\n C. 5\n D. 6\n",
        ),
        Message(role=Role.ASSISTANT, content="Answer:"),
    ]
    assert sample.ground_truth == " B"
    assert sample.possible_completions == [" A", " B", " C", " D"]


def test_mmlu_full_text_prompt() -> None:
    # Full text: options shown as a bulleted list; scores the full answer text.
    benchmark = mmlu_full_text(dataset=DatasetStub({"test": [_EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0)
    assert sample.messages == [
        Message(
            role=Role.USER,
            content="The following are multiple choice questions (with possible answers) about abstract algebra.\n"
            "Answer with the full text of the correct answer.\n\n"
            "Question: What is 2 + 2?\nPossible answers:\n- 3\n- 4\n- 5\n- 6\n",
        ),
        Message(role=Role.ASSISTANT, content="Answer:"),
    ]
    assert sample.ground_truth == " 4"
    assert sample.possible_completions == [" 3", " 4", " 5", " 6"]


def test_mmlu_idk_prompt() -> None:
    # IDK: an abstention option "?" is scored; the preamble names the subject by its raw (underscored) key.
    benchmark = mmlu_idk(dataset=DatasetStub({"test": [_EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0)
    assert sample.messages == [
        Message(
            role=Role.USER,
            content=f"The following are multiple choice questions (with answers) about {_SUBJECT}. "
            "Answer only if you are confident, since mistakes may be penalised, while correct answers receive "
            "points. It is acceptable to answer with '?' if you are unsure, and you will receive 0 points.\n\n"
            "Question: What is 2 + 2?\nA. 3\nB. 4\nC. 5\nD. 6\n",
        ),
        Message(role=Role.ASSISTANT, content="Answer:"),
    ]
    assert sample.ground_truth == " B"
    assert sample.possible_completions == [" A", " B", " C", " D", " ?"]


@pytest.mark.parametrize(
    "make_benchmark, display_name",
    [
        pytest.param(mmlu, "MMLU", id="mmlu"),
        pytest.param(mmlu_olmes, "MMLU_OLMES", id="mmlu_olmes"),
        pytest.param(mmlu_full_text, "Full Text MMLU", id="full_text"),
        pytest.param(mmlu_idk, "MMLU_IDK", id="mmlu_idk"),
    ],
)
def test_mmlu_display_name(make_benchmark: Callable[..., Benchmark], display_name: str) -> None:
    # Full Text MMLU keeps its spelled-out display name even though its registry identity is "FullTextMMLU".
    assert make_benchmark().display_name() == display_name
