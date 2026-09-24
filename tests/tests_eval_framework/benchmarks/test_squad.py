"""Specification of the composed SQuAD tasks.

``SQuAD_OLMES`` (v1) is free-form with an OLMES Title/Background layout; ``SQuAD2_MA`` and
``SQuAD2_MA_NO_SYSPROMPT`` (v2) share the MA prompt and differ only in whether the MA SYSTEM turn is present.
All score against several equally-correct gold spans. ``test_formatter_hash`` pins each against the real
HuggingFace data; the offline tests pin the assembled prompt (system turn included) against a stub dataset.
"""

from typing import Any

import pytest

from eval_framework.benchmarks.squad import (
    _MA_SYSTEM_PROMPT,
    _OLMES_PREAMBLE,
    SQUAD_BENCHMARKS,
    squad2_ma,
    squad2_ma_no_sysprompt,
    squad_olmes,
)
from eval_framework.contract import Benchmark
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter, Role
from tests.tests_eval_framework.benchmarks.utils import DatasetStub, first_sample
from tests.tests_eval_framework.tasks.benchmarks.utils import assert_benchmark_formatter_hash

# The old BaseTask test pinned only these two formatters; a stale (never-validated) NoStripConcat hash for the
# v2 variants means we keep the same scope here rather than rewrite recorded hashes.


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter])
@pytest.mark.parametrize("benchmark", SQUAD_BENCHMARKS, ids=lambda b: b.id())
def test_formatter_hash(benchmark: Benchmark, formatter_cls: type[BaseFormatter]) -> None:
    assert_benchmark_formatter_hash(benchmark, formatter_cls)


# ---------------------------------------------------------------------------
# Offline prompt spec
# ---------------------------------------------------------------------------

_V1_ROW: dict[str, Any] = {
    "title": "France",
    "context": "Paris is the capital of France.",
    "question": "What is the capital of France?",
    "answers": {"text": ["Paris"], "answer_start": [0]},
}

_V2_ROW: dict[str, Any] = {
    "context": "Paris is the capital of France.",
    "question": "What is the capital of France?",
    "answers": {"text": ["Paris"], "answer_start": [0]},
}


def test_squad_olmes_prompt() -> None:
    benchmark = squad_olmes(dataset=DatasetStub({"validation": [_V1_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0)
    assert sample.messages[0].role == Role.USER  # no system turn
    assert sample.messages[0].content == (
        f"{_OLMES_PREAMBLE}\n\n"
        "Title: France\nBackground: Paris is the capital of France.\nQuestion: What is the capital of France?\n"
    )
    assert sample.messages[1].role == Role.ASSISTANT and sample.messages[1].content == "Answer:"
    assert sample.ground_truth == [" Paris"]
    assert sample.possible_completions is None


def test_squad2_ma_prepends_the_system_prompt() -> None:
    benchmark = squad2_ma(dataset=DatasetStub({"validation": [_V2_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0)
    assert sample.messages[0].role == Role.SYSTEM and sample.messages[0].content == _MA_SYSTEM_PROMPT
    assert sample.messages[1].role == Role.USER
    assert (
        sample.messages[1].content
        == "Context:\nParis is the capital of France.\n\nQuestion:\nWhat is the capital of France?\n"
    )
    assert sample.ground_truth == ["Paris"]  # no leading space, several golds accepted


def test_squad2_ma_no_sysprompt_has_no_system_turn() -> None:
    benchmark = squad2_ma_no_sysprompt(dataset=DatasetStub({"validation": [_V2_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0)
    assert sample.messages[0].role == Role.USER  # no system turn
    assert all(m.role != Role.SYSTEM for m in sample.messages)
