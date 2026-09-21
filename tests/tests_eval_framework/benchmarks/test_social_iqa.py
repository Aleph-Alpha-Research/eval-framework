"""Specification of the Social IQa task.

The spec test builds the real benchmark (``social_iqa_mc_olmes``) over a fictional dataset and asserts the
assembled messages, ground truth, and scored completions — so this file reads as Social IQa's prompt spec,
with ``composed.py`` an implementation detail. Only the OLMES variant is registered. ``test_formatter_hash``
separately pins it against the real HuggingFace data.
"""

from typing import Any

import pytest

from eval_framework.benchmarks.social_iqa import SOCIAL_IQA_BENCHMARKS, social_iqa_mc_olmes
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
@pytest.mark.parametrize("benchmark", SOCIAL_IQA_BENCHMARKS, ids=lambda b: b.id())
def test_formatter_hash(benchmark: Benchmark, formatter_cls: type[BaseFormatter]) -> None:
    assert_benchmark_formatter_hash(benchmark, formatter_cls)


# ---------------------------------------------------------------------------
# Prompt spec: build the real benchmark over fictional rows, assert the assembled messages
# ---------------------------------------------------------------------------

# Fictional rows in the Social IQa format (NOT real dataset examples): a context + question, three answer
# options, and a 1-indexed label. Options are shown in fixed order (no shuffle). The shown question joins
# the context and the question with a single space.
_EVAL_ROW: dict[str, Any] = {
    "context": "Alex spilled coffee on a stranger.",
    "question": "What will Alex want to do next?",
    "answerA": "Apologise",
    "answerB": "Laugh",
    "answerC": "Walk away",
    "label": "1",
}
_FEWSHOT_ROW: dict[str, Any] = {
    "context": "Jordan studied all night for the exam.",
    "question": "How would Jordan feel afterwards?",
    "answerA": "Careless",
    "answerB": "Tired",
    "answerC": "Bored",
    "label": "2",
}

_EVAL_INSTRUCTION = (
    "Question: Alex spilled coffee on a stranger. What will Alex want to do next?\n"
    " A. Apologise\n B. Laugh\n C. Walk away\n"
)
_FEWSHOT_INSTRUCTION = (
    "Question: Jordan studied all night for the exam. How would Jordan feel afterwards?\n"
    " A. Careless\n B. Tired\n C. Bored\n"
)


def test_social_iqa_olmes_zeroshot_prompt() -> None:
    # Given the real Social IQa (OLMES) benchmark over one fictional row on its sample split (train)
    benchmark = social_iqa_mc_olmes(dataset=DatasetStub({"train": [_EVAL_ROW]}))
    # When we assemble its first sample (zero-shot)
    sample = first_sample(benchmark, num_fewshot=0)
    # Then the context+question is the shown question, options are space-prefixed letters, letters are scored:
    assert sample.messages == [
        Message(role=Role.USER, content=_EVAL_INSTRUCTION),
        Message(role=Role.ASSISTANT, content="Answer:"),
    ]
    assert sample.ground_truth == " A"
    assert sample.possible_completions == [" A", " B", " C"]


def test_social_iqa_olmes_oneshot_prepends_the_solved_demonstration() -> None:
    # Given the benchmark drawing its fewshot demonstration from the same split (train)
    benchmark = social_iqa_mc_olmes(dataset=DatasetStub({"train": [_FEWSHOT_ROW, _EVAL_ROW]}))
    # When we assemble its first sample (one-shot)
    sample = first_sample(benchmark, num_fewshot=1)
    # Then the solved demonstration (cue + correct letter) precedes the eval row's own zero-shot prompt:
    assert sample.messages == [
        Message(role=Role.USER, content=_FEWSHOT_INSTRUCTION),
        Message(role=Role.ASSISTANT, content="Answer: B"),
        Message(role=Role.USER, content=_EVAL_INSTRUCTION),
        Message(role=Role.ASSISTANT, content="Answer:"),
    ]
    assert sample.ground_truth == " A"
    assert sample.possible_completions == [" A", " B", " C"]
