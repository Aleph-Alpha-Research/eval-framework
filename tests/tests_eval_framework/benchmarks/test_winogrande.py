"""Specification of the Winogrande task.

The spec test builds the real benchmark (``winogrande_cloze``) over a fictional dataset and asserts the
assembled samples — so this file reads as Winogrande's prompt spec, with ``composed.py`` an implementation
detail. The registered task is partial evaluation: one item fans into two samples that score the shared
suffix under each option-augmented prefix. ``test_formatter_hash`` separately pins it against the real
HuggingFace data.
"""

from typing import Any

import pytest

from eval_framework.benchmarks.winogrande import winogrande_cloze
from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_names import register_winogrande_tasks
from template_formatting.formatter import (
    BaseFormatter,
    ConcatFormatter,
    Llama3Formatter,
    NoStripConcatFormatter,
)
from tests.tests_eval_framework.benchmarks.utils import DatasetStub
from tests.tests_eval_framework.tasks.benchmarks.utils import run_formatter_hash_test

# Registry for this test suite only holding the composed winogrande task.
_winogrande_registry = Registry()
register_winogrande_tasks(registry=_winogrande_registry)


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter, NoStripConcatFormatter])
@pytest.mark.parametrize("task_name", _winogrande_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, registry=_winogrande_registry)


# Fictional row in the Winogrande format (NOT a real dataset example): a sentence with a blank "_" filled
# by option1 or option2; "answer" selects the correct one (1-based).
_EVAL_ROW: dict[str, Any] = {
    "sentence": "The trophy doesn't fit in the suitcase because _ is too large.",
    "option1": "the trophy",
    "option2": "the suitcase",
    "answer": "1",
}


def test_winogrande_partial_eval_fans_one_item_into_a_scored_pair() -> None:
    # Given the winogrande (partial-eval) benchmark over a single fictional item on its sample split (train)
    benchmark = winogrande_cloze(dataset=DatasetStub({"train": [_EVAL_ROW]}))

    # When we assemble its samples (zero-shot)
    samples = list(benchmark.create(0, None, None, seed=42).iterate_samples())

    # Then one item becomes two samples scoring the same suffix under "prefix + option"; option1 is correct
    assert len(samples) == 2
    assert [s.ground_truth for s in samples] == ["True", "False"]
    assert all(s.possible_completions == [" is too large."] for s in samples)
    assert samples[0].messages[-1].content == "The trophy doesn't fit in the suitcase because the trophy"
    assert samples[1].messages[-1].content == "The trophy doesn't fit in the suitcase because the suitcase"
