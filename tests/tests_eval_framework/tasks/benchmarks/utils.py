"""Shared helpers for per-benchmark formatter hash tests."""

import random
import sys

import pytest

from eval_framework.tasks.registry import (
    _REGISTRY,
    TaskPlaceholder,
    get_task,
    registered_task_names,
)
from template_formatting.formatter import BaseFormatter
from tests.tests_eval_framework.utils import assert_hash_string


def _module_of_registered_task(task_name: str) -> str:
    task_key = _REGISTRY._task_key(task_name)
    _, value = _REGISTRY._registry[task_key]
    if isinstance(value, TaskPlaceholder):
        return value.module
    return value.__module__


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
