"""Specification of the German Winogrande (EllaMind) tasks.

Each spec test builds the real benchmark (via its ``winogrande_ellamind_*_de`` constructor) over a
fictional dataset and asserts the assembled samples — so this file reads as Winogrande's prompt spec,
with ``composed.py`` an implementation detail. The rows are fictional so this open source codebase does
not leak the real dataset into training data. Cloze/MC score the two full "option + suffix" completions;
partial evaluation fans one item into two samples that score the shared suffix under each option.
``test_formatter_hash`` separately pins the real benchmarks against the real HuggingFace data.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from eval_framework.benchmarks.winogrande_ellamind import (
    winogrande_ellamind_cloze_de,
    winogrande_ellamind_mc_de,
    winogrande_ellamind_partial_eval_de,
)
from eval_framework.contract import Benchmark
from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_names import register_winogrande_ellamind_tasks
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter, Message, Role
from tests.tests_eval_framework.benchmarks.utils import DatasetStub, first_sample
from tests.tests_eval_framework.tasks.benchmarks.utils import run_formatter_hash_test

# Registry for this test suite only holding winogrande_ellamind tasks
_winogrande_ellamind_registry = Registry()
register_winogrande_ellamind_tasks(registry=_winogrande_ellamind_registry)


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter])
@pytest.mark.parametrize("task_name", _winogrande_ellamind_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, registry=_winogrande_ellamind_registry)


# ---------------------------------------------------------------------------
# Prompt spec (Cloze / MC): build the real benchmark over fictional rows, assert the assembled messages
# ---------------------------------------------------------------------------

# Fictional rows (NOT real dataset examples). The blank ``_`` is filled by option1/option2; ``answer``
# selects the correct one. Choices are shuffled deterministically (seed: question + correct_answer).
_EVAL_ROW: dict[str, Any] = {
    "sentence": "Georg war größer als Johannes, daher konnte _ weiter sehen.",
    "option1": "Georg",
    "option2": "Johannes",
    "answer": "1",  # option1 (Georg) is correct
}
_FEWSHOT_ROW: dict[str, Any] = {
    "sentence": "Anna lief schneller als Maria, daher gewann _ das Rennen.",
    "option1": "Maria",
    "option2": "Anna",
    "answer": "2",  # option2 (Anna) is correct
}


@dataclass(frozen=True)
class _ExpectedPrompt:
    messages: list[Message]
    ground_truth: str
    possible_completions: list[str]


# --- Zero-shot ---
# Cloze: sentence-completion (no cue), scores the two full "option + suffix" strings.
_CLOZE = _ExpectedPrompt(
    messages=[Message(role=Role.USER, content="Georg war größer als Johannes, daher konnte")],
    ground_truth=" Georg weiter sehen.",
    possible_completions=[" Georg weiter sehen.", " Johannes weiter sehen."],
)
# MC: options listed, scores the letters.
_MC = _ExpectedPrompt(
    messages=[
        Message(
            role=Role.USER,
            content="Frage: Georg war größer als Johannes, daher konnte\n"
            "A. Georg weiter sehen.\nB. Johannes weiter sehen.\n",
        ),
        Message(role=Role.ASSISTANT, content="Antwort:"),
    ],
    ground_truth=" A",
    possible_completions=[" A", " B"],
)


@pytest.mark.parametrize(
    "make_benchmark, expected",
    [
        pytest.param(winogrande_ellamind_cloze_de, _CLOZE, id="cloze"),
        pytest.param(winogrande_ellamind_mc_de, _MC, id="mc"),
    ],
)
def test_winogrande_zeroshot_prompt(make_benchmark: Callable[..., Benchmark], expected: _ExpectedPrompt) -> None:
    benchmark = make_benchmark(dataset=DatasetStub({"validation": [_EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0)
    assert sample.messages == expected.messages
    assert sample.ground_truth == expected.ground_truth
    assert sample.possible_completions == expected.possible_completions


# --- One-shot: the few-shot example, then the eval row's own zero-shot prompt ---
_CLOZE_FEWSHOT_MESSAGES = [
    Message(role=Role.USER, content="Anna lief schneller als Maria, daher gewann"),
    Message(role=Role.ASSISTANT, content=" Anna das Rennen."),
]
_MC_FEWSHOT_MESSAGES = [
    Message(
        role=Role.USER,
        content="Frage: Anna lief schneller als Maria, daher gewann\nA. Maria das Rennen.\nB. Anna das Rennen.\n",
    ),
    Message(role=Role.ASSISTANT, content="Antwort: B"),
]


def _oneshot(fewshot_messages: list[Message], eval_expected: _ExpectedPrompt) -> _ExpectedPrompt:
    return _ExpectedPrompt(
        messages=[*fewshot_messages, *eval_expected.messages],
        ground_truth=eval_expected.ground_truth,
        possible_completions=eval_expected.possible_completions,
    )


@pytest.mark.parametrize(
    "make_benchmark, expected",
    [
        pytest.param(winogrande_ellamind_cloze_de, _oneshot(_CLOZE_FEWSHOT_MESSAGES, _CLOZE), id="cloze"),
        pytest.param(winogrande_ellamind_mc_de, _oneshot(_MC_FEWSHOT_MESSAGES, _MC), id="mc"),
    ],
)
def test_winogrande_oneshot_prompt(make_benchmark: Callable[..., Benchmark], expected: _ExpectedPrompt) -> None:
    benchmark = make_benchmark(dataset=DatasetStub({"validation": [_FEWSHOT_ROW, _EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=1)
    assert sample.messages == expected.messages
    assert sample.ground_truth == expected.ground_truth
    assert sample.possible_completions == expected.possible_completions


# ---------------------------------------------------------------------------
# Partial evaluation: one item fans into two paired samples scoring the shared suffix per option
# ---------------------------------------------------------------------------
def test_winogrande_partial_eval_fans_one_item_into_a_scored_pair() -> None:
    # Given the partial-eval benchmark over a single fictional item
    benchmark = winogrande_ellamind_partial_eval_de(dataset=DatasetStub({"validation": [_EVAL_ROW]}))

    # When we assemble its samples (zero-shot)
    samples = list(benchmark.create(0, None, None, seed=42).iterate_samples())

    # Then one item becomes two samples: same suffix scored, one option correct, prompt = "prefix + option"
    assert len(samples) == 2
    assert [s.ground_truth for s in samples].count("True") == 1
    assert [s.ground_truth for s in samples].count("False") == 1
    assert all(s.possible_completions == [" weiter sehen."] for s in samples)
    assert samples[0].messages[-1].content == "Georg war größer als Johannes, daher konnte Georg"
    assert samples[1].messages[-1].content == "Georg war größer als Johannes, daher konnte Johannes"


def test_winogrande_partial_eval_uses_consecutive_ids_so_pairs_group_by_id_over_two() -> None:
    # Given the partial-eval benchmark over two items
    benchmark = winogrande_ellamind_partial_eval_de(dataset=DatasetStub({"validation": [_EVAL_ROW, _FEWSHOT_ROW]}))

    # When we assemble all samples
    sample_ids = [s.id for s in benchmark.create(0, None, None, seed=42).iterate_samples()]

    # Then ids run consecutively across the fan-out, so PartialEvalAccuracy's id // 2 pairs each item
    assert sample_ids == [0, 1, 2, 3]
    assert [sample_id // 2 for sample_id in sample_ids] == [0, 0, 1, 1]
