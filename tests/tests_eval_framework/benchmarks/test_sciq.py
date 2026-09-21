"""Specification of the SciQ task.

The spec test builds the real benchmark (``sciq_mc_olmes``) over a fictional dataset and asserts the
assembled messages, ground truth, and scored completions — so this file reads as SciQ's prompt spec, with
``composed.py`` an implementation detail. Options are shuffled deterministically per item, so the expected
option order below is the shuffle's actual output for these rows. Only the OLMES variant is registered.
``test_formatter_hash`` separately pins it against the real HuggingFace data.
"""

from typing import Any

import pytest

from eval_framework.benchmarks.sciq import SCIQ_BENCHMARKS, sciq_mc_olmes
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
@pytest.mark.parametrize("benchmark", SCIQ_BENCHMARKS, ids=lambda b: b.id())
def test_formatter_hash(benchmark: Benchmark, formatter_cls: type[BaseFormatter]) -> None:
    assert_benchmark_formatter_hash(benchmark, formatter_cls)


# ---------------------------------------------------------------------------
# Prompt spec: build the real benchmark over fictional rows, assert the assembled messages
# ---------------------------------------------------------------------------

# Fictional rows in the SciQ format (NOT real dataset examples): a question, the correct answer, and three
# distractors. The four options are shuffled deterministically (seeded by question + correct answer); the
# expected instructions below reflect that shuffle's actual output.
_EVAL_ROW: dict[str, Any] = {
    "question": "What is the chemical symbol for water?",
    "correct_answer": "H2O",
    "distractor1": "O2",
    "distractor2": "CO2",
    "distractor3": "NaCl",
}
_FEWSHOT_ROW: dict[str, Any] = {
    "question": "What gas do plants absorb from the air?",
    "correct_answer": "carbon dioxide",
    "distractor1": "oxygen",
    "distractor2": "nitrogen",
    "distractor3": "hydrogen",
}

_EVAL_INSTRUCTION = "Question: What is the chemical symbol for water?\n A. CO2\n B. H2O\n C. O2\n D. NaCl\n"
_FEWSHOT_INSTRUCTION = (
    "Question: What gas do plants absorb from the air?\n A. carbon dioxide\n B. hydrogen\n C. nitrogen\n D. oxygen\n"
)


def test_sciq_olmes_zeroshot_prompt() -> None:
    # Given the real SciQ (OLMES) benchmark over one fictional row on its sample split (train)
    benchmark = sciq_mc_olmes(dataset=DatasetStub({"train": [_EVAL_ROW]}))
    # When we assemble its first sample (zero-shot)
    sample = first_sample(benchmark, num_fewshot=0)
    # Then the shuffled options are space-prefixed lettered choices, and the letters are scored:
    assert sample.messages == [
        Message(role=Role.USER, content=_EVAL_INSTRUCTION),
        Message(role=Role.ASSISTANT, content="Answer:"),
    ]
    assert sample.ground_truth == " B"  # H2O landed at position B
    assert sample.possible_completions == [" A", " B", " C", " D"]


def test_sciq_olmes_oneshot_draws_the_demonstration_from_the_same_split_without_leaking() -> None:
    # Given the benchmark whose fewshot split equals its sample split (train): the eval item is excluded
    # from its own demonstrations. Seed 42 puts the eval row first, leaving the fewshot row as the demo.
    benchmark = sciq_mc_olmes(dataset=DatasetStub({"train": [_FEWSHOT_ROW, _EVAL_ROW]}))
    # When we assemble its first sample (one-shot)
    sample = first_sample(benchmark, num_fewshot=1)
    # Then the solved demonstration (cue + correct letter) precedes the eval row's own zero-shot prompt:
    assert sample.messages == [
        Message(role=Role.USER, content=_FEWSHOT_INSTRUCTION),
        Message(role=Role.ASSISTANT, content="Answer: A"),  # carbon dioxide landed at position A
        Message(role=Role.USER, content=_EVAL_INSTRUCTION),
        Message(role=Role.ASSISTANT, content="Answer:"),
    ]
    assert sample.ground_truth == " B"
    assert sample.possible_completions == [" A", " B", " C", " D"]
