"""Specification of the composed HumanEvalPlus tasks.

HumanEvalPlus reuses HumanEval's ``_V2`` prompt shape on the EvalPlus dataset, so this pins the two variants
against the real data (0-shot, as their recorded hashes were taken) and checks the assembled prompt / gold
against a stub, confirming the reused builders behave identically on the new dataset.
"""

from typing import Any

import pytest

from eval_framework.benchmarks.humaneval_plus import HUMANEVAL_PLUS_BENCHMARKS, humaneval_plus, humaneval_plus_bpb
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
@pytest.mark.parametrize("benchmark", HUMANEVAL_PLUS_BENCHMARKS, ids=lambda b: b.id())
def test_formatter_hash(benchmark: Benchmark, formatter_cls: type[BaseFormatter]) -> None:
    assert_benchmark_formatter_hash(benchmark, formatter_cls, num_fewshot=0)  # recorded 0-shot


_ROW: dict[str, Any] = {
    "task_id": "HumanEval/0",
    "prompt": '\n\ndef add(a: int, b: int) -> int:\n    """Adds two numbers."""\n',
    "canonical_solution": "    return a + b",
    "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
    "entry_point": "add",
}


def test_execution_uses_v2_prompt_shape() -> None:
    benchmark = humaneval_plus(dataset=DatasetStub({"test": [_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0)
    assert sample.messages == [Message(role=Role.USER, content="```python\n" + _ROW["prompt"].rstrip() + "\n")]
    assert sample.ground_truth == "Success"
    assert sample.possible_completions is None


def test_bpb_scores_fenced_solution() -> None:
    benchmark = humaneval_plus_bpb(dataset=DatasetStub({"test": [_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0)
    gold = _ROW["canonical_solution"].rstrip() + "\n```"
    assert sample.messages == [Message(role=Role.USER, content="```python\n" + _ROW["prompt"].rstrip() + "\n")]
    assert sample.possible_completions == [gold]
    assert sample.ground_truth == gold
