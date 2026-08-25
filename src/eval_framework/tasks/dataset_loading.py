"""Fetching a benchmark's dataset, kept out of the composed classes so they stay free of Hugging Face
mechanics (revisions, cache directories, download config)."""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import cast

from datasets import DatasetDict, DownloadConfig, load_dataset


class DatasetLoader(ABC):
    """Loads a benchmark's dataset splits, selecting the subject config via ``name``."""

    @abstractmethod
    def load(self, name: str | None) -> DatasetDict: ...


class HfDatasetLoader(DatasetLoader):
    """Loads one Hugging Face dataset, pinned to ``revision``."""

    def __init__(self, dataset_path: str, revision: str | None) -> None:
        self._dataset_path = dataset_path
        self.revision = revision

    def load(self, name: str | None) -> DatasetDict:
        cache_dir = os.environ.get("HF_DATASET_CACHE_DIR", f"{Path.home()}/.cache/huggingface/datasets")
        download_config = DownloadConfig(cache_dir=cache_dir, max_retries=5)
        dataset = load_dataset(
            path=self._dataset_path,
            name=name,
            revision=self.revision,
            cache_dir=cache_dir,
            download_config=download_config,
        )
        return cast(DatasetDict, dataset)


class DatasetPolicy(ABC):
    """Produces the loader for a benchmark's dataset, and documents where that dataset comes from."""

    @abstractmethod
    def loader(self, custom_hf_revision: str | None) -> DatasetLoader: ...

    @abstractmethod
    def documentation(self) -> str:
        """Markdown for the task's ``## Dataset`` doc section, describing where the dataset comes from."""
        ...
