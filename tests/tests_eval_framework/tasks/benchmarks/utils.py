"""Shared helpers for per-benchmark formatter hash tests."""

import random
import sys
from dataclasses import dataclass
from unittest.mock import patch

import pytest
from datasets import Dataset, DatasetDict

from eval_framework.tasks.base import BaseTask, Sample
from eval_framework.tasks.registry import (
    _REGISTRY,
    registered_task_names,
    registry,
)
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Message
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

    def _instantiate(num_fewshot_value: int) -> object:
        task_instance = registry()[task_name].create(
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
            f"Failed to instantiate task {task_name=}: {e}; retrying with 0-shot",
            file=sys.stderr,
        )
        try:
            task_instance = _instantiate(0)
            sample = next(iter(task_instance.iterate_samples(1)))
        except Exception as inner:
            pytest.fail(f"Could not instantiate {task_name=}: {inner}")

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
        task_name=task_name,
        suffix_key=formatter_cls.__name__,
        tested_string=formatted_sample_with_completions,
    )


# ---------------------------------------------------------------------------
# Shared util functions for offline prompt tests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpectedPrompt:
    messages: list[Message]
    concat: str
    ground_truth: str | list[str] | None
    completions: list[str] | None


def _iterate_samples_over_mock_dataset(task: BaseTask, fictional_dataset: DatasetDict) -> Sample:
    """Same entry points as production: ``iterate_samples`` over a patched HF load."""
    with patch.object(task, "_load_hf_dataset", return_value=fictional_dataset):
        return next(iter(task.iterate_samples(1)))


def _assert_sample_matches(sample: Sample, expected: ExpectedPrompt) -> None:
    assert sample.messages == expected.messages
    assert ConcatFormatter().format(sample.messages, output_mode="string") == expected.concat
    assert sample.ground_truth == expected.ground_truth
    assert sample.possible_completions == expected.completions


def assert_offline_zeroshot_prompt(
    task_cls: type[BaseTask],
    eval_row: dict,
    *,
    subjects: list[str],
    expected: ExpectedPrompt,
) -> None:
    """Assert the 0-shot prompt. Only ``eval_row`` is needed with ``num_fewshot=0`` so a dataset
    with just the sample split suffices."""
    task = task_cls.with_overwrite(num_fewshot=0, custom_subjects=subjects, custom_hf_revision=None)
    mock_dataset = DatasetDict({task.SAMPLE_SPLIT: Dataset.from_list([eval_row])})
    _assert_sample_matches(_iterate_samples_over_mock_dataset(task, mock_dataset), expected)


def assert_offline_oneshot_prompt(
    task_cls: type[BaseTask],
    eval_row: dict,
    fewshot_row: dict,
    *,
    subjects: list[str],
    expected: ExpectedPrompt,
) -> None:
    """Assert the 1-shot prompt. The dataset layout depends on whether the task draws
    fewshot examples from a separate split (``FEWSHOT_SPLIT != SAMPLE_SPLIT``) or the same one."""
    task = task_cls.with_overwrite(num_fewshot=1, custom_subjects=subjects, custom_hf_revision=None)
    if task.FEWSHOT_SPLIT != task.SAMPLE_SPLIT:
        mock_dataset = DatasetDict(
            {
                task.SAMPLE_SPLIT: Dataset.from_list([eval_row]),
                task.FEWSHOT_SPLIT: Dataset.from_list([fewshot_row]),
            }
        )
    else:
        mock_dataset = DatasetDict(
            # Use fewshot row first such that after shuffling (with seed 42) the eval row is the first item
            {task.SAMPLE_SPLIT: Dataset.from_list([fewshot_row, eval_row])},
        )
    _assert_sample_matches(_iterate_samples_over_mock_dataset(task, mock_dataset), expected)
