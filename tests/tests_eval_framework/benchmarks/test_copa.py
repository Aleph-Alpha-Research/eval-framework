"""Specification of the COPA task.

The spec test builds the real benchmark (``copa_mc_olmes``) over a fictional dataset and asserts the
assembled messages, ground truth, and scored completions — so this file reads as COPA's prompt spec, with
``composed.py`` an implementation detail. The premise is recast as a sentence stem and the options continue
it directly (no assistant cue). Only the OLMES variant is registered. ``test_formatter_hash`` separately
pins it against the real HuggingFace data.
"""

from typing import Any

import pytest

from eval_framework.benchmarks.copa import copa_mc_olmes
from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_names import register_copa_tasks
from template_formatting.formatter import (
    BaseFormatter,
    ConcatFormatter,
    Llama3Formatter,
    Message,
    NoStripConcatFormatter,
    Role,
)
from tests.tests_eval_framework.benchmarks.utils import DatasetStub, first_sample
from tests.tests_eval_framework.tasks.benchmarks.utils import run_formatter_hash_test

# Registry for this test suite only holding the composed copa task.
_copa_registry = Registry()
register_copa_tasks(registry=_copa_registry)


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter, NoStripConcatFormatter])
@pytest.mark.parametrize("task_name", _copa_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, registry=_copa_registry)


# ---------------------------------------------------------------------------
# Prompt spec: build the real benchmark over fictional rows, assert the assembled messages
# ---------------------------------------------------------------------------

# Fictional rows in the COPA format (NOT real dataset examples): a premise, a "cause"/"effect" cue, two
# alternatives, and the 0-based index of the correct one. The premise's period becomes the connector and
# the alternatives are lower-cased to continue the stem.
_EVAL_ROW: dict[str, Any] = {
    "premise": "The man broke his toe.",
    "question": "cause",
    "choice1": "He dropped a hammer on his foot.",
    "choice2": "He got a hangnail.",
    "label": 0,
}
_FEWSHOT_ROW: dict[str, Any] = {
    "premise": "The woman won the race.",
    "question": "effect",
    "choice1": "She trained hard.",
    "choice2": "She received a trophy.",
    "label": 1,
}

_EVAL_INSTRUCTION = "The man broke his toe because\n A. he dropped a hammer on his foot.\n B. he got a hangnail.\n"
_FEWSHOT_INSTRUCTION = "The woman won the race therefore\n A. she trained hard.\n B. she received a trophy.\n"


def test_copa_olmes_zeroshot_prompt() -> None:
    # Given the real COPA (OLMES) benchmark over one fictional row on its sample split (validation)
    benchmark = copa_mc_olmes(dataset=DatasetStub({"validation": [_EVAL_ROW]}))
    # When we assemble its first sample (zero-shot)
    sample = first_sample(benchmark, num_fewshot=0)
    # Then the stem-plus-options is a lone USER turn (no assistant cue), and the letters are scored:
    assert sample.messages == [Message(role=Role.USER, content=_EVAL_INSTRUCTION)]
    assert sample.ground_truth == " A"
    assert sample.possible_completions == [" A", " B"]


def test_copa_olmes_oneshot_prepends_the_solved_demonstration() -> None:
    # Given the benchmark drawing its fewshot demonstration from a separate split (test)
    benchmark = copa_mc_olmes(dataset=DatasetStub({"validation": [_EVAL_ROW], "test": [_FEWSHOT_ROW]}))
    # When we assemble its first sample (one-shot)
    sample = first_sample(benchmark, num_fewshot=1)
    # Then the solved demonstration (the correct letter as its answer) precedes the eval row's own prompt:
    assert sample.messages == [
        Message(role=Role.USER, content=_FEWSHOT_INSTRUCTION),
        Message(role=Role.ASSISTANT, content=" B"),
        Message(role=Role.USER, content=_EVAL_INSTRUCTION),
    ]
    assert sample.ground_truth == " A"
    assert sample.possible_completions == [" A", " B"]
