"""Specification of the German ARC (EllaMind) tasks.

Each spec test builds the real benchmark over a fictional dataset and asserts the assembled sample — with
``composed.py`` an implementation detail. Each test injects the rows of a single subject directly; how the
two subjects are carved out of the shared "deu" config by the ``arc_config`` column is a ``SubjectColumn``
concern, covered in ``test_dataset_loading``. ``test_formatter_hash`` pins the tasks against real data.
"""

from typing import Any

import pytest

from eval_framework.benchmarks.arc_ellamind import (
    arc_ellamind_bpb_de,
    arc_ellamind_cloze_de,
    arc_ellamind_mc_de,
)
from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_names import register_arc_ellamind_tasks
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

# Registry for this test suite only holding the composed arc_ellamind tasks.
_arc_ellamind_registry = Registry()
register_arc_ellamind_tasks(registry=_arc_ellamind_registry)


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter, NoStripConcatFormatter])
@pytest.mark.parametrize("task_name", _arc_ellamind_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, registry=_arc_ellamind_registry)


# Fictional German ARC rows (NOT real dataset examples): each tagged with its arc_config subset; the
# ``choices`` field is a flat list, and ``answer_key`` is a letter.
_EASY_ROW: dict[str, Any] = {
    "arc_config": "ARC-Easy",
    "question": "Was ist H2O?",
    "choices": ["Wasser", "Feuer", "Luft", "Erde"],
    "answer_key": "A",
}
_HARD_ROW: dict[str, Any] = {
    "arc_config": "ARC-Challenge",
    "question": "Was ist die Ordnungszahl von Sauerstoff?",
    "choices": ["6", "7", "8", "9"],
    "answer_key": "C",
}


def test_arc_ellamind_cloze_zeroshot_prompt() -> None:
    # Given the cloze benchmark restricted to ARC-Easy over one fictional row
    benchmark = arc_ellamind_cloze_de(dataset=DatasetStub({"test": [_EASY_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0, custom_subjects=["ARC-Easy"])
    # Then the German cloze prompt scores the full answer texts:
    assert sample.messages == [
        Message(role=Role.USER, content="Frage: Was ist H2O?\n"),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ]
    assert sample.ground_truth == " Wasser"
    assert sample.possible_completions == [" Wasser", " Feuer", " Luft", " Erde"]


def test_arc_ellamind_mc_zeroshot_prompt() -> None:
    # Given the multiple-choice benchmark restricted to ARC-Challenge over one fictional row
    benchmark = arc_ellamind_mc_de(dataset=DatasetStub({"test": [_HARD_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0, custom_subjects=["ARC-Challenge"])
    # Then the options are lettered (not space-prefixed) and the letters are scored:
    assert sample.messages == [
        Message(role=Role.USER, content="Frage: Was ist die Ordnungszahl von Sauerstoff?\nA. 6\nB. 7\nC. 8\nD. 9\n"),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ]
    assert sample.ground_truth == " C"
    assert sample.possible_completions == [" A", " B", " C", " D"]


def test_arc_ellamind_bpb_scores_only_the_ground_truth() -> None:
    # BPB scores a single forward pass over the correct answer only.
    benchmark = arc_ellamind_bpb_de(dataset=DatasetStub({"test": [_EASY_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0, custom_subjects=["ARC-Easy"])
    assert sample.ground_truth == " Wasser"
    assert sample.possible_completions == [" Wasser"]
