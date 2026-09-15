"""Specification of the MedQA task.

The spec test builds the real benchmark (``medqa_mc_olmes``) over a fictional dataset and asserts the
assembled messages, ground truth, and scored completions — so this file reads as MedQA's prompt spec, with
``composed.py`` an implementation detail. Only the OLMES variant is registered. ``test_formatter_hash``
separately pins it against the real HuggingFace data.
"""

from typing import Any

import pytest

from eval_framework.benchmarks.medqa import medqa_mc_olmes
from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_names import register_medqa_tasks
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

# Registry for this test suite only holding the composed medqa task.
_medqa_registry = Registry()
register_medqa_tasks(registry=_medqa_registry)


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter, NoStripConcatFormatter])
@pytest.mark.parametrize("task_name", _medqa_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, registry=_medqa_registry)


# ---------------------------------------------------------------------------
# Prompt spec: build the real benchmark over fictional rows, assert the assembled messages
# ---------------------------------------------------------------------------

# Fictional rows in the MedQA format (NOT real dataset examples): an exam question, five options, and the
# 0-based index of the correct one. Options are shown in fixed order (no shuffle).
_EVAL_ROW: dict[str, Any] = {
    "question": "What is the powerhouse of the cell?",
    "choices": ["Nucleus", "Mitochondrion", "Ribosome", "Golgi apparatus", "Lysosome"],
    "answer_idx": 1,
}
_FEWSHOT_ROW: dict[str, Any] = {
    "question": "Which vitamin is synthesised in the skin?",
    "choices": ["Vitamin A", "Vitamin B12", "Vitamin C", "Vitamin D", "Vitamin K"],
    "answer_idx": 3,
}

_EVAL_INSTRUCTION = (
    "Question: What is the powerhouse of the cell?\n"
    " A. Nucleus\n B. Mitochondrion\n C. Ribosome\n D. Golgi apparatus\n E. Lysosome\n"
)
_FEWSHOT_INSTRUCTION = (
    "Question: Which vitamin is synthesised in the skin?\n"
    " A. Vitamin A\n B. Vitamin B12\n C. Vitamin C\n D. Vitamin D\n E. Vitamin K\n"
)


def test_medqa_olmes_zeroshot_prompt() -> None:
    # Given the real MedQA (OLMES) benchmark over one fictional row on its sample split (test)
    benchmark = medqa_mc_olmes(dataset=DatasetStub({"test": [_EVAL_ROW]}))
    # When we assemble its first sample (zero-shot)
    sample = first_sample(benchmark, num_fewshot=0)
    # Then options are space-prefixed lettered choices, and the letters are scored:
    assert sample.messages == [
        Message(role=Role.USER, content=_EVAL_INSTRUCTION),
        Message(role=Role.ASSISTANT, content="Answer:"),
    ]
    assert sample.ground_truth == " B"
    assert sample.possible_completions == [" A", " B", " C", " D", " E"]


def test_medqa_olmes_oneshot_prepends_the_solved_demonstration() -> None:
    # Given the benchmark drawing its fewshot demonstration from a separate split (train)
    benchmark = medqa_mc_olmes(dataset=DatasetStub({"test": [_EVAL_ROW], "train": [_FEWSHOT_ROW]}))
    # When we assemble its first sample (one-shot)
    sample = first_sample(benchmark, num_fewshot=1)
    # Then the solved demonstration (cue + correct letter) precedes the eval row's own zero-shot prompt:
    assert sample.messages == [
        Message(role=Role.USER, content=_FEWSHOT_INSTRUCTION),
        Message(role=Role.ASSISTANT, content="Answer: D"),
        Message(role=Role.USER, content=_EVAL_INSTRUCTION),
        Message(role=Role.ASSISTANT, content="Answer:"),
    ]
    assert sample.ground_truth == " B"
    assert sample.possible_completions == [" A", " B", " C", " D", " E"]
