"""Test helpers for offline prompt tests and formatter hash pinning tests.

A benchmark's dataset is an injected policy, so a test can build the real benchmark over a fictional
in-memory dataset and assert its assembled messages — no download, and ``composed.py`` stays an
implementation detail.

Hash helpers use the real Hugging Face data and compare the hash of the formatted prompt plus its completions and ground
truth against ``task-prompts-hashes.json`` to indicate prompt changes.
"""

import random
import sys
from typing import Any, final, override

import pytest
from datasets import Dataset, DatasetDict

from eval_framework.contract import Benchmark, Sample
from eval_framework.tasks.dataset_loading import DatasetLoader, DatasetPolicy
from template_formatting.formatter import BaseFormatter
from tests.tests_eval_framework.utils import assert_hash_string


@final
class DatasetStub(DatasetPolicy, DatasetLoader):
    """A fictional in-memory dataset, injected in place of a benchmark's pinned Hugging Face policy."""

    def __init__(self, splits: dict[str, list[dict[str, Any]]]) -> None:
        self._splits = splits

    @override
    def loader(self, custom_hf_revision: str | None) -> DatasetLoader:
        return self

    @override
    def documentation(self) -> str:
        return "fictional in-memory dataset"

    @override
    def load(self, name: str | None) -> DatasetDict:
        return DatasetDict({split: Dataset.from_list(rows) for split, rows in self._splits.items()})

    @override
    def metadata(self) -> dict[str, str]:
        return {"dataset_path": "stub"}


def first_sample(benchmark: Benchmark, *, num_fewshot: int, custom_subjects: list[str] | None = None) -> Sample:
    """Build the benchmark's eval (seed 42) and return its first assembled sample."""
    evaluation = benchmark.create(
        num_fewshot=num_fewshot, custom_subjects=custom_subjects, custom_hf_revision=None, seed=42
    )
    return next(iter(evaluation.iterate_samples(1)))


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


def assert_benchmark_formatter_hash(
    benchmark: Benchmark, formatter_cls: type[BaseFormatter], *, num_fewshot: int = 1
) -> None:
    """Pin one benchmark x formatter against its recorded hash, keyed by ``benchmark.id()``. No registry: the
    caller parametrises over the benchmark objects directly."""
    sample = _sample_for_hash(benchmark, num_fewshot=num_fewshot)
    assert_hash_string(
        task_name=benchmark.id(),
        suffix_key=formatter_cls.__name__,
        tested_string=_hash_payload(sample, formatter_cls),
    )


def _sample_for_hash(benchmark: Benchmark, *, num_fewshot: int) -> Sample:
    """First assembled sample (full HF data, seed 42), retrying 0-shot if the requested shot count can't
    be created."""
    _seed_for_determinism()
    try:
        instance = benchmark.create(num_fewshot=num_fewshot, custom_subjects=None, custom_hf_revision=None, seed=42)
        return next(iter(instance.iterate_samples(1)))
    except Exception as e:
        label = repr(benchmark.id())
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
