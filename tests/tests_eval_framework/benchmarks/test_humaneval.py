"""Specification of the composed HumanEval tasks.

The ``_OLMES`` variants generate a function body scored by execution; the ``BPB`` variants score the gold
solution's loglikelihood as a single candidate. ``test_formatter_hash`` pins each variant against the real
HuggingFace data (at the shot count its recorded hash was taken with); the offline tests pin the assembled
prompt / candidates / context against a stub dataset, and ``test_reconstruct_program`` pins the runnable
program handed to the execution metric.
"""

from typing import Any

import pytest

from eval_framework.benchmarks.humaneval import (
    _OLMES_STOP_SEQUENCES,
    CODE_TO_EXECUTE,
    HUMANEVAL_BENCHMARKS,
    HumanEvalMetricContext,
    _reconstruct_program,
    humaneval_bpb,
    humaneval_bpb_v2,
    humaneval_olmes,
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

# The shot count each variant's recorded hash was taken with (the OLMES 3-shot recommendation; everything
# else at the harness default of 1).
_NUM_FEWSHOT = {
    "HumanEval_OLMES": 3,
    "HumanEval_OLMES_V2": 1,
    "HumanEvalBPB": 1,
    "HumanEvalBPB_V2": 1,
}


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter, NoStripConcatFormatter])
@pytest.mark.parametrize("benchmark", HUMANEVAL_BENCHMARKS, ids=lambda b: b.id())
def test_formatter_hash(benchmark: Benchmark, formatter_cls: type[BaseFormatter]) -> None:
    assert_benchmark_formatter_hash(benchmark, formatter_cls, num_fewshot=_NUM_FEWSHOT[benchmark.id()])


# ---------------------------------------------------------------------------
# Offline prompt / candidate / context specs
# ---------------------------------------------------------------------------

# A fictional row in the HumanEval format (the real prompt column often starts with blank lines).
_ROW: dict[str, Any] = {
    "task_id": "HumanEval/0",
    "prompt": '\n\ndef add(a: int, b: int) -> int:\n    """Adds two numbers."""\n',
    "canonical_solution": "    return a + b",
    "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
    "entry_point": "add",
}


def test_olmes_execution_prompt_and_context() -> None:
    # Free-form generation: the open code fence with the stub, no assistant cue, a placeholder gold, and the
    # test harness carried as the per-sample context the execution metric runs against.
    benchmark = humaneval_olmes(dataset=DatasetStub({"test": [_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0)
    assert sample.messages == [Message(role=Role.USER, content="```python\n" + _ROW["prompt"])]
    assert sample.ground_truth == "Success"
    assert sample.possible_completions is None
    assert sample.context == HumanEvalMetricContext(test=_ROW["test"], entry_point="add", prompt=_ROW["prompt"])


def test_bpb_v2_scores_fenced_solution() -> None:
    benchmark = humaneval_bpb_v2(dataset=DatasetStub({"test": [_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0)
    gold = _ROW["canonical_solution"].rstrip() + "\n```"
    assert sample.messages == [Message(role=Role.USER, content="```python\n" + _ROW["prompt"].rstrip() + "\n")]
    assert sample.possible_completions == [gold]
    assert sample.ground_truth == gold


def test_bpb_v1_scores_space_prefixed_solution() -> None:
    benchmark = humaneval_bpb(dataset=DatasetStub({"test": [_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0)
    assert sample.messages == [Message(role=Role.USER, content="```python\n" + _ROW["prompt"].lstrip())]
    assert sample.possible_completions == [" " + _ROW["canonical_solution"]]
    assert sample.ground_truth == " " + _ROW["canonical_solution"]


def test_reconstruct_program_splices_and_truncates() -> None:
    # The generation runs past the solution into a new statement; reconstruction truncates at a stop sequence
    # before splicing the body into the runnable program the execution metric runs.
    assert "\nprint" in _OLMES_STOP_SEQUENCES
    context = HumanEvalMetricContext(test=_ROW["test"], entry_point="add", prompt="def add(a: int, b: int) -> int:\n")
    program = _reconstruct_program(
        "    return a + b\nprint('debug')", context=context, ground_truth="Success", messages=[]
    )
    assert program == CODE_TO_EXECUTE.format(
        start_of_code=context.prompt,
        completion_text="    return a + b",
        test_code=context.test,
        entry_point=context.entry_point,
    )
