"""Test helpers for specifying composed benchmarks offline.

A benchmark's dataset is an injected policy, so a test can build the real benchmark over a fictional
in-memory dataset and assert its assembled messages — no download, and ``composed.py`` stays an
implementation detail.
"""

from typing import Any, final, override

from datasets import Dataset, DatasetDict

from eval_framework.contract import Benchmark, Sample
from eval_framework.tasks.dataset_loading import DatasetLoader, DatasetPolicy


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
