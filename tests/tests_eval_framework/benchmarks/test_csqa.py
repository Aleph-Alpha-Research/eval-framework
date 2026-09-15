"""Specification of the CommonsenseQA task.

The spec test builds the real benchmark (``csqa_mc_olmes``) over a fictional dataset and asserts the
assembled messages, ground truth, and scored completions — so this file reads as CommonsenseQA's prompt
spec, with ``composed.py`` an implementation detail. Only the OLMES variant is registered.
``test_formatter_hash`` separately pins it against the real HuggingFace data.
"""

from typing import Any

import pytest

from eval_framework.benchmarks.csqa import csqa_mc_olmes
from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_names import register_csqa_tasks
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

# Registry for this test suite only holding the composed csqa task.
_csqa_registry = Registry()
register_csqa_tasks(registry=_csqa_registry)


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter, NoStripConcatFormatter])
@pytest.mark.parametrize("task_name", _csqa_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, registry=_csqa_registry)


# ---------------------------------------------------------------------------
# Prompt spec: build the real benchmark over fictional rows, assert the assembled messages
# ---------------------------------------------------------------------------

# Fictional rows in the CommonsenseQA format (NOT real dataset examples): a question, five labelled options
# (the nested ``choices`` mapping), and the ``answerKey`` label. Options are shown in fixed order (no shuffle).
_EVAL_ROW: dict[str, Any] = {
    "question": "Where would you store a pillow you are not using?",
    "choices": {"label": ["A", "B", "C", "D", "E"], "text": ["bedroom", "cupboard", "floor", "kitchen", "car"]},
    "answerKey": "B",
}
_FEWSHOT_ROW: dict[str, Any] = {
    "question": "What do people do when they are tired?",
    "choices": {"label": ["A", "B", "C", "D", "E"], "text": ["run", "sleep", "eat", "sing", "swim"]},
    "answerKey": "B",
}

_EVAL_INSTRUCTION = (
    "Question: Where would you store a pillow you are not using?\n"
    " A. bedroom\n B. cupboard\n C. floor\n D. kitchen\n E. car\n"
)
_FEWSHOT_INSTRUCTION = (
    "Question: What do people do when they are tired?\n A. run\n B. sleep\n C. eat\n D. sing\n E. swim\n"
)


def test_csqa_olmes_zeroshot_prompt() -> None:
    # Given the real CommonsenseQA (OLMES) benchmark over one fictional row on its sample split (train)
    benchmark = csqa_mc_olmes(dataset=DatasetStub({"train": [_EVAL_ROW]}))
    # When we assemble its first sample (zero-shot)
    sample = first_sample(benchmark, num_fewshot=0)
    # Then options are space-prefixed lettered choices, and the letters are scored:
    assert sample.messages == [
        Message(role=Role.USER, content=_EVAL_INSTRUCTION),
        Message(role=Role.ASSISTANT, content="Answer:"),
    ]
    assert sample.ground_truth == " B"
    assert sample.possible_completions == [" A", " B", " C", " D", " E"]


def test_csqa_olmes_oneshot_draws_the_demonstration_from_the_same_split_without_leaking() -> None:
    # Given the benchmark whose fewshot split equals its sample split (train): the eval item is excluded
    # from its own demonstrations. Seed 42 puts the eval row first, leaving the fewshot row as the demo.
    benchmark = csqa_mc_olmes(dataset=DatasetStub({"train": [_FEWSHOT_ROW, _EVAL_ROW]}))
    # When we assemble its first sample (one-shot)
    sample = first_sample(benchmark, num_fewshot=1)
    # Then the solved demonstration (cue + correct letter) precedes the eval row's own zero-shot prompt:
    assert sample.messages == [
        Message(role=Role.USER, content=_FEWSHOT_INSTRUCTION),
        Message(role=Role.ASSISTANT, content="Answer: B"),
        Message(role=Role.USER, content=_EVAL_INSTRUCTION),
        Message(role=Role.ASSISTANT, content="Answer:"),
    ]
    assert sample.ground_truth == " B"
    assert sample.possible_completions == [" A", " B", " C", " D", " E"]
