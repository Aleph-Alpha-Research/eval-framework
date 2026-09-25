"""Specification of the composed MBPP tasks.

The ``_OLMES`` / ``_EvalPlus`` variants generate a function scored by running the gold asserts; the ``BPB``
variants score the gold solution's loglikelihood. ``test_formatter_hash`` pins each against the real data (at
the recorded shot count); the offline tests pin the assembled prompt / candidates and the assert-appending
reconstruction against a stub dataset.
"""

from typing import Any

import pytest

from eval_framework.benchmarks.mbpp import (
    MBPP_BENCHMARKS,
    MBPP_DATASET_PATH,
    _code_expander,
    instruct_reconstruct,
    mbpp_bpb,
    mbpp_evalplus,
    mbpp_olmes,
    olmes_reconstruct,
)
from eval_framework.contract import Benchmark
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.utils import run_python_code
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

# MBPP_OLMES's recorded hash is 3-shot; the rest were recorded at the harness default of 1 (the fixed 3-shot
# EvalPlus block ignores the requested count anyway).
_NUM_FEWSHOT = {"MBPP_OLMES": 3}


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter, NoStripConcatFormatter])
@pytest.mark.parametrize("benchmark", MBPP_BENCHMARKS, ids=lambda b: b.id())
def test_formatter_hash(benchmark: Benchmark, formatter_cls: type[BaseFormatter]) -> None:
    assert_benchmark_formatter_hash(benchmark, formatter_cls, num_fewshot=_NUM_FEWSHOT.get(benchmark.id(), 1))


# ---------------------------------------------------------------------------
# Offline prompt / candidate / reconstruction specs
# ---------------------------------------------------------------------------

_ROW: dict[str, Any] = {
    "text": "Write a function to add two numbers.",
    "code": "def add(a, b):\n    return a + b",
    "test_list": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"],
}


def test_olmes_prompt() -> None:
    # MBPP_OLMES uses a fixed 3-shot block, so the eval item is the last USER turn + the assistant cue.
    sample = first_sample(mbpp_olmes(dataset=DatasetStub({"test": [_ROW]})), num_fewshot=3)
    assert sample.messages[-2] == Message(
        role=Role.USER,
        content=(
            "Please provide a self-contained Python script that solves the following problem in a markdown "
            "code block:\n```\nWrite a function to add two numbers.\nassert add(1, 2) == 3\n```\n"
        ),
    )
    assert sample.messages[-1] == Message(role=Role.ASSISTANT, content="Here is the completed function:\n\n```python\n")
    assert sample.ground_truth == "['assert add(1, 2) == 3', 'assert add(0, 0) == 0']"
    assert sample.possible_completions is None


def test_evalplus_prompt() -> None:
    sample = first_sample(mbpp_evalplus(dataset=DatasetStub({"test": [_ROW]})), num_fewshot=0)
    assert sample.messages[-1] == Message(
        role=Role.ASSISTANT,
        content=(
            "Below is a Python script with a self-contained function that solves the problem"
            " and passes corresponding tests:\n```python"
        ),
    )


def test_bpb_scores_space_prefixed_solution() -> None:
    sample = first_sample(mbpp_bpb(dataset=DatasetStub({"test": [_ROW], "train": [_ROW]})), num_fewshot=0)
    assert sample.messages[-1] == Message(role=Role.ASSISTANT, content="```python")  # the BEGIN cue
    assert sample.possible_completions == [" " + _ROW["code"]]
    assert sample.ground_truth == " " + _ROW["code"]


def test_olmes_reconstruct_appends_asserts() -> None:
    # The reconstruction truncates at a stop sequence, then appends the gold asserts wrapped to print a score.
    program = olmes_reconstruct(
        "def add(a, b):\n    return a + b\n```\ngarbage",
        context=None,
        ground_truth="['assert add(1, 2) == 3']",
        messages=[],
    )
    # Truncated at "```" (keeping its trailing newline), then a newline + the wrapped assert block are appended.
    assert program == (
        "def add(a, b):\n    return a + b\n\n"
        "try:\n    assert add(1, 2) == 3\n    score = True\nexcept:\n    score = False\nprint(score)"
    )


@pytest.mark.parametrize(
    "code, gt_asserts, expected",
    [
        # Valid asserts: each is indented under a try/except that sets score and prints it.
        (
            "def func(x):\n    return x + 1\n",
            "['assert func(1) == 2', 'assert func(3) == 4']",
            "def func(x):\n    return x + 1\ntry:\n    assert func(1) == 2\n    assert func(3) == 4\n"
            "    score = True\nexcept:\n    score = False\nprint(score)",
        ),
        # Empty list: still wraps in try/except (no asserts inside).
        (
            "def func(x):\n    return x + 1\n",
            "[]",
            "def func(x):\n    return x + 1\ntry:\n    score = True\nexcept:\n    score = False\nprint(score)",
        ),
        # No ground truth at all: code returned unchanged.
        ("def func(x):\n    return x + 1\n", "", "def func(x):\n    return x + 1\n"),
        # Not a list (a bare string): code returned unchanged.
        ("def func(x):\n    return x + 1\n", "'assert func(1) == 2'", "def func(x):\n    return x + 1\n"),
        # Multiline code is preserved verbatim before the appended asserts.
        (
            "def func(x):\n    if x < 0:\n        return 0\n    return x * 2\n",
            "['assert func(-1) == 0', 'assert func(2) == 4']",
            "def func(x):\n    if x < 0:\n        return 0\n    return x * 2\ntry:\n    assert func(-1) == 0\n"
            "    assert func(2) == 4\n    score = True\nexcept:\n    score = False\nprint(score)",
        ),
        # Several asserts are each indented on their own line inside the try block.
        (
            "def func(x):\n    return x * 2\n",
            "['assert func(1) == 2', 'assert func(2) == 4', 'assert func(0) == 0']",
            "def func(x):\n    return x * 2\ntry:\n    assert func(1) == 2\n    assert func(2) == 4\n"
            "    assert func(0) == 0\n    score = True\nexcept:\n    score = False\nprint(score)",
        ),
    ],
)
def test_code_expander(code: str, gt_asserts: str, expected: str) -> None:
    assert _code_expander(code, gt_asserts) == expected


def test_instruct_reconstruct_extracts_code_block() -> None:
    # The instruct variant's model answers in free-form prose; the code block is pulled out of the response
    # before the gold asserts are appended (used by MBPPDEEvalPlusInstruct).
    program = instruct_reconstruct(
        "Hier ist die Lösung:\n```python\ndef zwei():\n    return 2\n```",
        context=None,
        ground_truth="['assert zwei() == 2']",
        messages=[],
    )
    assert program.startswith("def zwei():\n    return 2\n")
    assert "Hier ist die Lösung" not in program
    assert "assert zwei() == 2" in program


def test_canonical_solutions_execute_end_to_end() -> None:
    # End-to-end smoke over the real dataset: expand each canonical solution with its gold asserts and run it
    # in the sandbox, confirming it passes. Reconstruction and the executing metric are covered separately
    # (the reconstruct tests above and metrics/test_code_assertion.py); this pins their composition on real data.
    # _code_expander is called directly because in real use the LLM output is stop-truncated before reconstruction,
    # and a canonical solution may itself contain a stop sequence (a comment or assert) that would corrupt it.
    dataset = pinned_by_framework(MBPP_DATASET_PATH).loader(None).load("full")
    for i in range(5):
        item = dataset["test"][i]
        code = _code_expander(item["code"] + "\n", str(item["test_list"]))
        result = run_python_code(code)
        assert result.endswith("True"), f"Item {i} failed: {result}"
