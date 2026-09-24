"""Specification of the composed IFEval tasks.

IFEval is 0-shot instruction-following: the prompt carries verifiable constraints, there is no gold answer,
and grading runs from a per-sample ``IFEvalMetricContext``. ``test_formatter_hash`` pins each variant against
the real data (0-shot); the offline test pins the assembled prompt, the ``None`` ground truth, and the
kwargs-normalised context against a stub dataset.
"""

from typing import Any

import pytest

from eval_framework.benchmarks.ifeval import IFEVAL_BENCHMARKS, ifeval
from eval_framework.contract import Benchmark
from eval_framework.metrics.completion.ifeval import IFEvalMetricContext
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
@pytest.mark.parametrize("benchmark", IFEVAL_BENCHMARKS, ids=lambda b: b.id())
def test_formatter_hash(benchmark: Benchmark, formatter_cls: type[BaseFormatter]) -> None:
    assert_benchmark_formatter_hash(benchmark, formatter_cls, num_fewshot=0)  # IFEval is 0-shot only


_ROW: dict[str, Any] = {
    "key": 1,
    "prompt": "Write a poem about the sea. Your answer must contain exactly 3 bullet points.",
    "instruction_id_list": ["detectable_format:number_bullet_lists"],
    # A float-typed integer kwarg (normalised to int) and a None kwarg (dropped).
    "kwargs": [{"num_bullets": 3.0, "relation": None}],
}


def test_ifeval_prompt_context_and_no_gold() -> None:
    benchmark = ifeval(dataset=DatasetStub({"train": [_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0)
    # 0-shot, empty cue: a single USER turn carrying the prompt verbatim.
    assert sample.messages == [Message(role=Role.USER, content=_ROW["prompt"])]
    assert sample.ground_truth is None
    assert sample.possible_completions is None
    assert sample.context == IFEvalMetricContext(
        key=1,
        instruction_id_list=["detectable_format:number_bullet_lists"],
        prompt=_ROW["prompt"],
        additional_kwargs=[{"num_bullets": 3}],  # 3.0 -> 3, None kwarg dropped
    )
