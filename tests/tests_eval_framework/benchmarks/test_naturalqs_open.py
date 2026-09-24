"""Specification of the composed Natural Questions (open) tasks.

``NaturalQsOpen`` generates a free-form answer scored by DROP F1 / exact match — carrying every gold answer as
a per-sample ``DropMetricContext`` and a list ground truth — while ``NaturalQsOpenMC_OLMES`` scores labelled
candidates by loglikelihood. ``test_formatter_hash`` pins both against the real HuggingFace data; the offline
tests pin the assembled prompt (and the context) against a fictional in-memory dataset.
"""

from typing import Any

import pytest

from eval_framework.benchmarks.naturalqs_open import (
    NATURALQS_OPEN_BENCHMARKS,
    natural_qs_open,
    natural_qs_open_mc_olmes,
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
@pytest.mark.parametrize("benchmark", NATURALQS_OPEN_BENCHMARKS, ids=lambda b: b.id())
def test_formatter_hash(benchmark: Benchmark, formatter_cls: type[BaseFormatter]) -> None:
    assert_benchmark_formatter_hash(benchmark, formatter_cls)


# ---------------------------------------------------------------------------
# Offline prompt spec
# ---------------------------------------------------------------------------

_OPEN_ROW: dict[str, Any] = {"question": "what is the capital of france", "answer": ["Paris", "paris"]}

_CHOICE_ROW: dict[str, Any] = {
    "question": "what is the capital of france",
    "choices": {"text": ["Paris", "Berlin", "Rome"], "label": ["A", "B", "C"]},
    "answerKey": "A",
}


def test_natural_qs_open_prompt_and_context() -> None:
    # Free-form: the question prompt, an "Answer:" cue, the list of gold answers, and every gold answer handed
    # to the F1 metric as a per-sample context.
    benchmark = natural_qs_open(dataset=DatasetStub({"validation": [_OPEN_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0)
    assert sample.messages[-2].role == Role.USER
    assert sample.messages[-2].content == "Question: what is the capital of france\n"
    assert sample.messages[-1].role == Role.ASSISTANT and sample.messages[-1].content == "Answer:"
    assert sample.ground_truth == [" Paris", " paris"]
    assert sample.possible_completions is None  # free-form generation, no scored candidates
    assert sample.context == DropMetricContext(answer_tuples=[["Paris"], ["paris"]])


def test_natural_qs_open_mc_olmes_prompt() -> None:
    # OLMES lays out the options with a leading space and scores the letter labels.
    benchmark = natural_qs_open_mc_olmes(dataset=DatasetStub({"validation": [_CHOICE_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0)
    assert sample.messages[-2].content == ("Question: what is the capital of france\n A. Paris\n B. Berlin\n C. Rome\n")
    assert sample.possible_completions == [" A", " B", " C"]
    assert sample.ground_truth == " A"
