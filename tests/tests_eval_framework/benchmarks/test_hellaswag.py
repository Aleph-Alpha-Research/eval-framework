"""Specification of the HellaSwag tasks.

Each spec test builds the real benchmark over a fictional dataset and asserts the assembled sample — so
this file reads as HellaSwag's prompt spec, with ``composed.py`` an implementation detail. Both variants
render identically (sentence-completion cloze) and differ only in which split is evaluated.
``test_formatter_hash`` separately pins them against the real HuggingFace data.
"""

from collections.abc import Callable
from typing import Any

import pytest

from eval_framework.benchmarks.hellaswag import HellaswagReader, hellaswag, hellaswag_olmes
from eval_framework.contract import Benchmark
from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_names import register_hellaswag_tasks
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

_hellaswag_registry = Registry()
register_hellaswag_tasks(registry=_hellaswag_registry)


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter, NoStripConcatFormatter])
@pytest.mark.parametrize("task_name", _hellaswag_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, registry=_hellaswag_registry)


# ---------------------------------------------------------------------------
# Prompt spec: build the real benchmark over fictional rows, assert the assembled sample
# ---------------------------------------------------------------------------

# Fictional rows in the HellaSwag format (NOT real dataset examples): an activity label, a two-part
# context (ctx_a + a capitalised ctx_b), candidate endings, and the correct ending's index.
_EVAL_ROW: dict[str, Any] = {
    "activity_label": "Cleaning windows",
    "ctx_a": "A woman stands at a tall window holding a spray bottle and a cloth.",
    "ctx_b": "she",
    "endings": [
        "sprays the glass and wipes it until it is clear.",
        "drinks the cleaning liquid straight from the bottle.",
        "throws the cloth out of the open window.",
        "sets the spray bottle on fire.",
    ],
    "label": "0",
}
_FEWSHOT_ROW: dict[str, Any] = {
    "activity_label": "Making tea",
    "ctx_a": "A man places a kettle of water on the stove.",
    "ctx_b": "he",
    "endings": [
        "waits for it to boil, then pours it over a tea bag.",
        "puts the kettle in the freezer to warm it up.",
    ],
    "label": "0",
}

# The shown text is "{activity}: {ctx_a} {Ctx_b}" (ctx_b capitalised), scored as a sentence continuation.
_EVAL_QUESTION = "Cleaning windows: A woman stands at a tall window holding a spray bottle and a cloth. She"
_FEWSHOT_QUESTION = "Making tea: A man places a kettle of water on the stove. He"
_GROUND_TRUTH = " sprays the glass and wipes it until it is clear."
_COMPLETIONS = [
    " sprays the glass and wipes it until it is clear.",
    " drinks the cleaning liquid straight from the bottle.",
    " throws the cloth out of the open window.",
    " sets the spray bottle on fire.",
]


@pytest.mark.parametrize(
    "make_benchmark, sample_split",
    [
        pytest.param(hellaswag, "validation", id="hellaswag"),
        pytest.param(hellaswag_olmes, "train", id="hellaswag_olmes"),
    ],
)
def test_hellaswag_zeroshot_prompt(make_benchmark: Callable[..., Benchmark], sample_split: str) -> None:
    # Given the real benchmark over one fictional row (on the variant's own sample split)
    benchmark = make_benchmark(dataset=DatasetStub({sample_split: [_EVAL_ROW]}))
    # When we assemble its first sample (zero-shot)
    sample = first_sample(benchmark, num_fewshot=0)
    # Then the prompt is a single user turn (no cue), scoring the full endings:
    assert sample.messages == [Message(role=Role.USER, content=_EVAL_QUESTION)]
    assert sample.ground_truth == _GROUND_TRUTH
    assert sample.possible_completions == _COMPLETIONS


def test_hellaswag_oneshot_prompt() -> None:
    # Given the benchmark with a fewshot row (train split) and the eval row (validation split)
    benchmark = hellaswag(dataset=DatasetStub({"validation": [_EVAL_ROW], "train": [_FEWSHOT_ROW]}))
    # When we assemble its first sample (one-shot)
    sample = first_sample(benchmark, num_fewshot=1)
    # Then the fewshot example (rendered with its correct ending) precedes the eval prompt:
    assert sample.messages == [
        Message(role=Role.USER, content=_FEWSHOT_QUESTION),
        Message(role=Role.ASSISTANT, content=" waits for it to boil, then pours it over a tea bag."),
        Message(role=Role.USER, content=_EVAL_QUESTION),
    ]
    assert sample.ground_truth == _GROUND_TRUTH
    assert sample.possible_completions == _COMPLETIONS


def test_hellaswag_reader_strips_markup_and_collapses_spaces() -> None:
    # Given an item whose text carries wikiHow markup and doubled spaces
    fields = HellaswagReader().read(
        {
            "activity_label": "Baking",
            "ctx_a": "Preheat [header] the  oven.",
            "ctx_b": "then bake",
            "endings": ["wait ten minutes.", "melt the pan."],
            "label": "0",
        }
    )
    # Then bracketed markup is removed, doubled spaces collapse, and ctx_b is capitalised
    assert fields.raw_question == "Baking: Preheat the oven. Then bake"
    assert fields.choices == ["wait ten minutes.", "melt the pan."]
    assert fields.correct_index == 0
