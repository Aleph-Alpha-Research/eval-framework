"""Shared helpers for per-benchmark formatter hash tests."""

import random
import sys
from dataclasses import dataclass
from unittest.mock import patch

import pytest
from datasets import Dataset, DatasetDict

from eval_framework.contract import Benchmark
from eval_framework.tasks.base import BaseTask, Sample
from eval_framework.tasks.registry import Registry
from eval_framework.tasks.registry import registry as global_registry
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Message
from tests.tests_eval_framework.utils import assert_hash_string


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


def _sample_for_hash(benchmark: Benchmark, *, label: str, num_fewshot: int) -> Sample:
    """First assembled sample (full HF data, seed 42), retrying 0-shot if the requested shot count can't
    be created."""
    _seed_for_determinism()
    try:
        instance = benchmark.create(num_fewshot=num_fewshot, custom_subjects=None, custom_hf_revision=None, seed=42)
        return next(iter(instance.iterate_samples(1)))
    except Exception as e:
        print(f"Failed to instantiate {label}: {e}; retrying with 0-shot", file=sys.stderr)
        try:
            instance = benchmark.create(num_fewshot=0, custom_subjects=None, custom_hf_revision=None, seed=42)
            return next(iter(instance.iterate_samples(1)))
        except Exception as inner:
            pytest.fail(f"Could not instantiate {label}: {inner} (with {num_fewshot}-shot it failed with: {e})")


def _hash_payload(sample: Sample, formatter_cls: type[BaseFormatter]) -> str:
    """The exact string hashed for a sample: the formatted prompt plus its completions and ground truth."""
    formatted_sample = formatter_cls().format(sample.messages, output_mode="string")

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

    return (
        f"{formatted_sample}\n\nPossible completion:\n{possible_completions_str}\n\nGround truth:\n{ground_truth_str}"
    )


def assert_benchmark_formatter_hash(
    benchmark: Benchmark, formatter_cls: type[BaseFormatter], *, num_fewshot: int = 1
) -> None:
    """Pin one benchmark x formatter against its recorded hash, keyed by ``benchmark.id()``. No registry: the
    caller parametrises over the benchmark objects directly."""
    sample = _sample_for_hash(benchmark, label=repr(benchmark.id()), num_fewshot=num_fewshot)
    assert_hash_string(
        task_name=benchmark.id(),
        suffix_key=formatter_cls.__name__,
        tested_string=_hash_payload(sample, formatter_cls),
    )


def run_formatter_hash_test(
    task_name: str, formatter_cls: type[BaseFormatter], num_fewshot: int = 1, registry: Registry | None = None
) -> None:
    """Name-based variant for tasks that must be resolved through a registry (e.g. still-lazy BaseTask
    benchmarks). Composed benchmarks use ``assert_benchmark_formatter_hash`` with the object directly."""
    registry = registry if registry is not None else global_registry()
    sample = _sample_for_hash(registry[task_name], label=f"task_name={task_name!r}", num_fewshot=num_fewshot)
    assert_hash_string(
        task_name=task_name,
        suffix_key=formatter_cls.__name__,
        tested_string=_hash_payload(sample, formatter_cls),
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
