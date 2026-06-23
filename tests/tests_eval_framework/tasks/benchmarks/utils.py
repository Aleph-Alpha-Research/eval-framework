"""Shared helpers for per-benchmark formatter hash tests."""

import random
import sys
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest
from datasets import Dataset, DatasetDict

from eval_framework.tasks.base import BaseTask, Sample
from eval_framework.tasks.registry import (
    _REGISTRY,
    get_task,
    registered_task_names,
)
from template_formatting.formatter import BaseFormatter, Message
from tests.tests_eval_framework.utils import assert_hash_string


def _module_of_registered_task(task_name: str) -> str:
    task_key = _REGISTRY._task_key(task_name)
    _, factory = _REGISTRY._registry[task_key]
    return factory.source_module


def get_task_names_for_module(module_name: str, skip_tasks: list[str] | None = None) -> list[str]:
    """Return registered eval-framework task names declared in a given benchmark module.

    Mirrors `eval_framework_companion.tests.tasks.benchmarks.utils.get_task_names_for_module`
    so per-benchmark test files can parametrize over just the tasks they own.
    """
    target_module = f"eval_framework.tasks.benchmarks.{module_name}"
    skip = set(skip_tasks or [])
    return sorted(
        name
        for name in registered_task_names()
        if _module_of_registered_task(name) == target_module and name not in skip
    )


def _seed_for_determinism() -> None:
    random.seed(42)
    try:
        import numpy as np

        np.random.seed(42)
    except ImportError:
        pass
    try:
        import datasets

        datasets.set_random_seed(42)
    except (ImportError, AttributeError):
        pass


def run_formatter_hash_test(task_name: str, formatter_cls: type[BaseFormatter], num_fewshot: int = 1) -> None:
    """Run the formatter hash consistency test for a single task x formatter combination.

    Uses the full HuggingFace datasets with seed 42 and a deterministic few-shot sampler,
    matching the prior `test_all_formatters.py` behaviour so existing hashes remain valid.
    """
    _seed_for_determinism()

    task_class = get_task(task_name)

    def _instantiate(num_fewshot_value: int) -> object:
        task_instance = task_class.with_overwrite(
            num_fewshot=num_fewshot_value,
            custom_subjects=None,
            custom_hf_revision=None,
        )
        original_method = task_instance._sample_fewshot_examples

        def deterministic_fewshot(item: dict) -> list:
            task_instance.rnd = random.Random(42)
            return original_method(item)

        task_instance._sample_fewshot_examples = deterministic_fewshot
        return task_instance

    try:
        task_instance = _instantiate(num_fewshot)
        sample = next(iter(task_instance.iterate_samples(1)))
    except Exception as e:
        print(
            f"Failed to instantiate task {task_class.__name__}: {e}; retrying with 0-shot",
            file=sys.stderr,
        )
        try:
            task_instance = _instantiate(0)
            sample = next(iter(task_instance.iterate_samples(1)))
        except Exception as inner:
            pytest.fail(f"Could not instantiate {task_class.__name__}: {inner}")

    formatter = formatter_cls()
    formatted_sample = formatter.format(sample.messages, output_mode="string")

    possible_completions = sample.possible_completions
    ground_truth = sample.ground_truth

    if possible_completions:
        possible_completions_str = "\n".join(f'- "{item}"' for item in possible_completions)
    else:
        possible_completions_str = "None"

    if ground_truth:
        if isinstance(ground_truth, list):
            ground_truth_str = "\n".join(f'- "{item}"' for item in ground_truth)
        else:
            ground_truth_str = f'- "{ground_truth}"'
    else:
        ground_truth_str = "None"

    formatted_sample_with_completions = (
        f"{formatted_sample}\n\nPossible completion:\n{possible_completions_str}\n\nGround truth:\n{ground_truth_str}"
    )

    assert_hash_string(
        task_name=task_class.__name__,
        suffix_key=formatter_cls.__name__,
        tested_string=formatted_sample_with_completions,
    )


# ---------------------------------------------------------------------------
# Declarative framework for offline prompt-assembly tests
# ---------------------------------------------------------------------------
#
# These tests verify the ONE thing each task is responsible for: the list of
# ``Message``s it assembles (plus ``ground_truth`` and ``possible_completions``).
#
# Rendering those messages into a final prompt string is the *formatter's* job and
# is already covered exhaustively in
# ``tests/tests_template_formatting/test_formatter_eval.py`` (ConcatFormatter,
# Llama3Formatter, cue/prefilling, whitespace, multi-turn, ...). Re-asserting the
# concatenated string here would just re-test the formatter and couple these tests
# to formatter implementation details, so we deliberately do NOT do it.
#
# Adding coverage for a new task is declarative -- see ``OfflinePromptCase``. The data is
# OPTIONAL: by default you only declare the ``columns`` the task reads, and the framework
# synthesizes a deterministic template row for them (see ``template_row``). The framework
# then drives the real production path (``iterate_samples``) with only the HuggingFace
# download patched out.
#
# How much the *content* of the data matters depends on the task:
#
#   * Passthrough tasks -- the row text flows verbatim into the messages, so the data has no
#     sanctity (it could be lorem ipsum). Just pass ``columns`` and let the template row be
#     generated; the test pins assembly structure: roles, ordering, intro line, prefixes,
#     cue-as-assistant-turn, few-shot interleaving. Build the expected messages by calling
#     ``template_value(column)`` so you never hard-code the placeholder text.
#
#   * Transformation tasks -- the row is *processed* before it becomes messages (e.g. gsm8k
#     strips ``<<...>>`` calculator spans from the question and rewrites the answer via
#     ``normalize_answer_str``: drop the ``#### N`` tail, collapse whitespace, capitalize,
#     append "So the answer is N.", space out operators). Template data would not exercise any
#     of that, so pass explicit ``eval_row``/``fewshot_row``: the row is the *input* and
#     ``ExpectedPrompt`` is the *transformed output*, with the row carrying the markers the
#     processing acts on (``<<...>>``, ``####``, bare operators like ``2+3``, ...). Heavy or
#     standalone transforms are also worth covering with direct unit tests (cf. the
#     ``_find_closing_bracket`` / ``_split_text_command`` tests in test_math_reasoning.py).


def template_value(column: str, *, variant: str = "eval") -> str:
    """Deterministic placeholder text for a passthrough column in template mode.

    Recognizable and unique per ``(column, variant)`` so that expected messages can be built
    by calling this same helper instead of hard-coding the placeholder text. ``variant``
    distinguishes the eval row from the few-shot row so they are visibly different in a prompt.
    """
    return f"lorem ipsum {column} ({variant})"


def template_row(columns: list[str], *, variant: str = "eval") -> dict[str, str]:
    """Synthesize a fictional row for the given column names (see ``template_value``)."""
    return {column: template_value(column, variant=variant) for column in columns}


def _resolve_row(row: dict[str, Any] | None, columns: list[str] | None, *, variant: str) -> dict[str, Any]:
    if row is not None:
        return row
    if not columns:
        raise ValueError("Provide an explicit row or `columns` so a template row can be synthesized.")
    return template_row(columns, variant=variant)


@dataclass(frozen=True)
class ExpectedPrompt:
    """Expected output of a task's sample construction (the task's own responsibility)."""

    messages: list[Message]
    ground_truth: str | list[str] | None
    completions: list[str] | None


def _iterate_samples_over_mock_dataset(task: BaseTask, fictional_dataset: DatasetDict) -> Sample:
    """Same entry point as production: ``iterate_samples`` over a patched HF load."""
    with patch.object(task, "_load_hf_dataset", return_value=fictional_dataset):
        return next(iter(task.iterate_samples(1)))


def _assert_sample_matches(sample: Sample, expected: ExpectedPrompt) -> None:
    assert sample.messages == expected.messages
    assert sample.ground_truth == expected.ground_truth
    assert sample.possible_completions == expected.completions


def assert_offline_zeroshot_prompt(
    task_cls: type[BaseTask],
    eval_row: dict[str, Any] | None = None,
    *,
    columns: list[str] | None = None,
    subjects: list[str] | None = None,
    expected: ExpectedPrompt,
) -> None:
    """Assert the 0-shot messages. Pass ``eval_row`` for transformation tasks, or ``columns``
    to fall back to a synthesized template row. With ``num_fewshot=0`` a dataset with just the
    sample split suffices."""
    task = task_cls.with_overwrite(num_fewshot=0, custom_subjects=subjects, custom_hf_revision=None)
    row = _resolve_row(eval_row, columns, variant="eval")
    mock_dataset = DatasetDict({task.SAMPLE_SPLIT: Dataset.from_list([row])})
    _assert_sample_matches(_iterate_samples_over_mock_dataset(task, mock_dataset), expected)


def assert_offline_oneshot_prompt(
    task_cls: type[BaseTask],
    eval_row: dict[str, Any] | None = None,
    fewshot_row: dict[str, Any] | None = None,
    *,
    columns: list[str] | None = None,
    subjects: list[str] | None = None,
    expected: ExpectedPrompt,
) -> None:
    """Assert the 1-shot messages. Pass explicit ``eval_row``/``fewshot_row`` for transformation
    tasks, or ``columns`` to fall back to synthesized template rows (the few-shot row uses a
    different ``variant`` so it is distinguishable from the eval row).

    The single few-shot example is injected by overriding ``_sample_fewshot_examples`` to
    return the few-shot row directly. This keeps the test independent of each task's sampling
    strategy (separate vs. shared split, predefined example pools, RNG seeding), so the SAME
    helper works for every task -- here we only care about prompt assembly, not sampling.
    """
    task = task_cls.with_overwrite(num_fewshot=1, custom_subjects=subjects, custom_hf_revision=None)
    eval_data = _resolve_row(eval_row, columns, variant="eval")
    fewshot_data = _resolve_row(fewshot_row, columns, variant="fewshot")
    mock_dataset = DatasetDict({task.SAMPLE_SPLIT: Dataset.from_list([eval_data])})
    # A fresh copy per call: _get_example_messages mutates the row (sets ``subject``).
    task._sample_fewshot_examples = lambda item, _row=fewshot_data: [dict(_row)]  # type: ignore[method-assign]
    _assert_sample_matches(_iterate_samples_over_mock_dataset(task, mock_dataset), expected)


@dataclass(frozen=True)
class OfflinePromptCase:
    """Declarative spec for a task's offline prompt-assembly test.

    To cover a new task, append one of these to the module's ``CASES`` list:
      * ``task_cls``    -- the task class under test
      * ``zeroshot``    -- expected messages/ground_truth/completions with ``num_fewshot=0``
      * ``columns``     -- the dataset columns the task reads. In template mode (no explicit
                           ``eval_row``) the framework synthesizes the row data from these names;
                           build the expected messages with ``template_value(column)``.
      * ``eval_row``    -- OPTIONAL explicit row, overriding template generation. Needed for
                           transformation tasks: it is the *input* and must carry the markers
                           the task processes; the expected messages hold the *processed* output.
      * ``fewshot_row`` -- OPTIONAL explicit few-shot row (falls back to a template row when
                           ``columns`` is given and ``oneshot`` is set).
      * ``oneshot``     -- expected messages/ground_truth/completions with ``num_fewshot=1``
      * ``subjects``    -- restrict to these subjects (defaults to the task's own ``SUBJECTS``)

    Provide either ``columns`` or ``eval_row`` (or both -- explicit rows win).
    """

    task_cls: type[BaseTask]
    zeroshot: ExpectedPrompt
    columns: list[str] | None = None
    eval_row: dict[str, Any] | None = None
    fewshot_row: dict[str, Any] | None = None
    oneshot: ExpectedPrompt | None = None
    subjects: list[str] | None = None

    def __post_init__(self) -> None:
        if self.eval_row is None and not self.columns:
            raise ValueError(f"{self.test_id}: provide `columns` (template mode) or an explicit `eval_row`.")

    @property
    def test_id(self) -> str:
        return self.task_cls.__name__


def run_offline_prompt_case(case: OfflinePromptCase) -> None:
    """Run the 0-shot (and, when specified, 1-shot) assembly assertions for one task."""
    assert_offline_zeroshot_prompt(
        case.task_cls,
        case.eval_row,
        columns=case.columns,
        subjects=case.subjects,
        expected=case.zeroshot,
    )
    if case.oneshot is not None:
        assert_offline_oneshot_prompt(
            case.task_cls,
            case.eval_row,
            case.fewshot_row,
            columns=case.columns,
            subjects=case.subjects,
            expected=case.oneshot,
        )
