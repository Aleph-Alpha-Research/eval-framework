"""Specification of the composed DROP tasks.

``DropCompletion_OLMES`` generates a free-form answer scored by DROP F1 / exact match — carrying the gold
answer tuples as a per-sample ``DropMetricContext`` — while ``DropMC_OLMES`` scores labelled candidates by
loglikelihood. ``test_formatter_hash`` pins both against the real HuggingFace data; the offline tests pin the
assembled prompt (and the context) against a fictional in-memory dataset.
"""

from typing import Any

import pytest

from eval_framework.benchmarks.drop import (
    _OLMES_PREAMBLE,
    DROP_BENCHMARKS,
    drop_completion_olmes,
    drop_mc_olmes,
)
from eval_framework.contract import Benchmark
from eval_framework.metrics.completion.drop_completion import DropMetricContext
from template_formatting.formatter import (
    BaseFormatter,
    ConcatFormatter,
    Llama3Formatter,
    NoStripConcatFormatter,
    Role,
)
from tests.tests_eval_framework.benchmarks.utils import DatasetStub, first_sample
from tests.tests_eval_framework.tasks.benchmarks.utils import assert_benchmark_formatter_hash


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter, NoStripConcatFormatter])
@pytest.mark.parametrize("benchmark", DROP_BENCHMARKS, ids=lambda b: b.id())
def test_formatter_hash(benchmark: Benchmark, formatter_cls: type[BaseFormatter]) -> None:
    assert_benchmark_formatter_hash(benchmark, formatter_cls)


# ---------------------------------------------------------------------------
# Offline prompt spec
# ---------------------------------------------------------------------------

_COMPLETION_ROW: dict[str, Any] = {
    "passage": "Paris is the capital of France.",
    "question": "What is the capital of France?",
    "answer": {"number": "", "spans": ["Paris"], "date": {"day": "", "month": "", "year": ""}},
    "validated_answers": {"number": [], "spans": [], "date": []},
}

_CHOICE_ROW: dict[str, Any] = {
    "passage_original": "Paris is the capital of France.",
    "question_original": "What is the capital?",
    "choices": {"text": ["Paris", "Berlin", "Rome"], "label": ["A", "B", "C"]},
    "answerKey": "B",
}


def test_drop_completion_olmes_prompt_and_context() -> None:
    # Free-form: the reading-comprehension preamble, the passage/question prompt, an "Answer:" cue for the
    # model to continue, and the gold answer tuples handed to the F1 metric as a per-sample context.
    benchmark = drop_completion_olmes(dataset=DatasetStub({"validation": [_COMPLETION_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0)
    assert sample.messages[0].role == Role.USER
    assert sample.messages[0].content == (
        f"{_OLMES_PREAMBLE}\n\nPassage: Paris is the capital of France.\nQuestion: What is the capital of France?\n"
    )
    assert sample.messages[1].role == Role.ASSISTANT and sample.messages[1].content == "Answer:"
    assert sample.ground_truth == " Paris"
    assert sample.possible_completions is None  # free-form generation, no scored candidates
    assert sample.context == DropMetricContext(answer_tuples=[["Paris"]])


def test_drop_mc_olmes_prompt() -> None:
    # OLMES lays out the options with a leading space and scores the letter labels.
    benchmark = drop_mc_olmes(dataset=DatasetStub({"validation": [_CHOICE_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0)
    assert sample.messages[-2].content == (
        "Passage: Paris is the capital of France.\nQuestion: What is the capital?\n A. Paris\n B. Berlin\n C. Rome\n"
    )
    assert sample.possible_completions == [" A", " B", " C"]
    assert sample.ground_truth == " B"
