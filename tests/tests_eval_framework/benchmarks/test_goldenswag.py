"""Specification of the GoldenSwag tasks.

GoldenSwag is a curated HellaSwag subset, read identically (see ``test_hellaswag``); these tests focus on
what GoldenSwag adds — the ``GoldenSwag_IDK`` variant, which lets the model abstain with an "I do not
know" answer, prefaced by a confidence preamble. ``test_formatter_hash`` pins both variants against the
real HuggingFace data.
"""

from typing import Any

import pytest

from eval_framework.benchmarks.goldenswag import _IDK_PREAMBLE, goldenswag, goldenswag_idk
from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_names import register_goldenswag_tasks
from template_formatting.formatter import (
    BaseFormatter,
    ConcatFormatter,
    Llama3Formatter,
    Message,
    NoStripConcatFormatter,
    Role,
)
from tests.tests_eval_framework.benchmarks.utils import DatasetStub, first_sample

_goldenswag_registry = Registry()
register_goldenswag_tasks(registry=_goldenswag_registry)


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter, NoStripConcatFormatter])
@pytest.mark.parametrize("task_name", _goldenswag_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    from tests.tests_eval_framework.tasks.benchmarks.utils import run_formatter_hash_test

    run_formatter_hash_test(task_name, formatter_cls, registry=_goldenswag_registry)


# Fictional row in the HellaSwag/GoldenSwag format (NOT a real dataset example).
_EVAL_ROW: dict[str, Any] = {
    "activity_label": "Cleaning windows",
    "ctx_a": "A woman stands at a tall window holding a spray bottle and a cloth.",
    "ctx_b": "she",
    "endings": [
        "sprays the glass and wipes it until it is clear.",
        "drinks the cleaning liquid straight from the bottle.",
    ],
    "label": "0",
}

_QUESTION = "Cleaning windows: A woman stands at a tall window holding a spray bottle and a cloth. She"
_ENDINGS = [
    " sprays the glass and wipes it until it is clear.",
    " drinks the cleaning liquid straight from the bottle.",
]


def test_goldenswag_zeroshot_prompt() -> None:
    # Given the base benchmark over one fictional row
    benchmark = goldenswag(dataset=DatasetStub({"validation": [_EVAL_ROW]}))
    # When we assemble its first sample (zero-shot)
    sample = first_sample(benchmark, num_fewshot=0)
    # Then it is a plain sentence-completion cloze, scoring the full endings:
    assert sample.messages == [Message(role=Role.USER, content=_QUESTION)]
    assert sample.ground_truth == _ENDINGS[0]
    assert sample.possible_completions == _ENDINGS


def test_goldenswag_idk_prepends_preamble_and_scores_an_abstention_option() -> None:
    # Given the IDK benchmark over one fictional row
    benchmark = goldenswag_idk(dataset=DatasetStub({"validation": [_EVAL_ROW]}))
    # When we assemble its first sample (zero-shot)
    sample = first_sample(benchmark, num_fewshot=0)
    # Then the confidence preamble prefaces the prompt, and "I do not know." is scored alongside the endings
    assert sample.messages == [Message(role=Role.USER, content=f"{_IDK_PREAMBLE}\n\n{_QUESTION}")]
    assert sample.ground_truth == _ENDINGS[0]
    assert sample.possible_completions == [*_ENDINGS, " I do not know."]
