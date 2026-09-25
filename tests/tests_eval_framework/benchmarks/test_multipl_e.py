"""Specification of the composed MultiPL-E tasks (12 per-language HumanEval/MBPP translations).

``test_formatter_hash`` pins each variant against the real HuggingFace data (0-shot — MultiPL-E has no
few-shot); the offline tests pin the assembled prompt / null gold / execution context and the stop-token trim.
"""

from typing import Any

import pytest

from eval_framework.benchmarks.multipl_e import (
    MULTIPL_E_BENCHMARKS,
    MULTIPL_E_STOP_TOKENS,
    _multipl_e,
    _trim_at_stops,
)
from eval_framework.contract import Benchmark
from eval_framework.metrics.completion.multipl_e_assertion import MultiPLEMetricContext
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
@pytest.mark.parametrize("benchmark", MULTIPL_E_BENCHMARKS, ids=lambda b: b.id())
def test_formatter_hash(benchmark: Benchmark, formatter_cls: type[BaseFormatter]) -> None:
    assert_benchmark_formatter_hash(benchmark, formatter_cls, num_fewshot=0)  # MultiPL-E is 0-shot only


_ROW: dict[str, Any] = {
    "prompt": "// Add two numbers\nint add(int a, int b) {\n",
    "tests": "int main() { assert(add(1, 2) == 3); }",
    "language": "cpp",
}


def test_prompt_and_context() -> None:
    benchmark = _multipl_e(
        "MultiPLEHumanEvalCpp", prefix="humaneval", lang="cpp", dataset=DatasetStub({"test": [_ROW]})
    )
    sample = first_sample(benchmark, num_fewshot=0)
    assert sample.messages == [Message(role=Role.USER, content=_ROW["prompt"])]  # the stub, verbatim, no cue
    assert sample.ground_truth is None
    assert sample.possible_completions is None
    assert sample.context == MultiPLEMetricContext(prompt=_ROW["prompt"], tests=_ROW["tests"], language="cpp")


def test_trim_at_stops() -> None:
    trim = _trim_at_stops(MULTIPL_E_STOP_TOKENS["cpp"])
    # cpp stops at "\n}"; everything after (a stray extra line) is dropped.
    assert trim("  return a + b;\n}\nint unrelated() {}") == "  return a + b;"
