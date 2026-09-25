"""Fetching a benchmark's dataset, kept out of the composed classes so they stay free of Hugging Face
mechanics (revisions, cache directories, download config)."""

import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, final, override

from datasets import DatasetDict, DownloadConfig, load_dataset


def _load_hf_dataset(dataset_path: str, revision: str | None, hf_config: str | None) -> DatasetDict:
    """Load ``dataset_path`` at ``revision`` from Hugging Face, selecting the ``hf_config`` config
    (``None`` = the dataset's default config)."""
    cache_dir = os.environ.get("HF_DATASET_CACHE_DIR", f"{Path.home()}/.cache/huggingface/datasets")
    download_config = DownloadConfig(cache_dir=cache_dir, max_retries=5)
    return load_dataset(
        path=dataset_path, name=hf_config, revision=revision, cache_dir=cache_dir, download_config=download_config
    )


class DatasetLoader(ABC):
    """Loads a benchmark's dataset splits for one subject."""

    @abstractmethod
    def load(self, name: str | None) -> DatasetDict:
        """Load the splits for subject ``name`` (``None`` = the task's single unnamed slice). Each loader decides
        how a subject selects data — ``HfDatasetLoader`` treats the subject as the HF config."""

    @abstractmethod
    def metadata(self) -> dict[str, str]:
        """Dataset-identifying metadata merged into the eval's ``get_metadata`` (e.g. the dataset path)."""
        ...


@final
class HfDatasetLoader(DatasetLoader):
    """Loads one Hugging Face dataset pinned to ``revision``, treating the requested subject as the HF config."""

    def __init__(self, dataset_path: str, revision: str | None) -> None:
        self.dataset_path = dataset_path
        self.revision = revision

    @override
    def metadata(self) -> dict[str, str]:
        return {"dataset_path": self.dataset_path}

    @override
    def load(self, name: str | None) -> DatasetDict:
        # ``name`` is the subject; for a plain HF dataset the subject *is* the config to load.
        return _load_hf_dataset(self.dataset_path, self.revision, hf_config=name)


@final
class FixedHfConfigLoader(DatasetLoader):
    """Loads a fixed HF config, ignoring the requested subject — for a dataset whose subjects are labels rather
    than configs (or a task with no subjects but a non-default config)."""

    def __init__(self, dataset_path: str, revision: str | None, hf_config: str | None) -> None:
        self._dataset_path = dataset_path
        self._revision = revision
        self._hf_config = hf_config

    @override
    def metadata(self) -> dict[str, str]:
        return {"dataset_path": self._dataset_path}

    @override
    def load(self, name: str | None) -> DatasetDict:
        return _load_hf_dataset(self._dataset_path, self._revision, hf_config=self._hf_config)


class DatasetPolicy(ABC):
    """Produces the loader for a benchmark's dataset, and documents where that dataset comes from."""

    @abstractmethod
    def loader(self, custom_hf_revision: str | None) -> DatasetLoader: ...

    @abstractmethod
    def documentation(self) -> str:
        """Markdown for the task's ``## Dataset`` doc section, describing where the dataset comes from."""
        ...

    def subset(self, keep: Callable[[dict[str, Any]], bool], description: str | None = None) -> "Subset":
        """Restrict this policy's dataset to the rows for which ``keep`` returns true (see ``Subset``).

        ``description`` names the resulting subset for the rendered dataset docs (e.g. "the diamond subset").
        """
        return Subset(self, keep, description)

    def subject_encoded_in_column(self, config: str | None, column: str) -> "SubjectColumn":
        """By default the loaded config names the subject; call this when the subject is instead encoded in
        a ``column`` of a single ``config`` (``None`` loads the default config) (see ``SubjectColumn``)."""
        return SubjectColumn(self, config, column)


@final
class _SubsetLoader(DatasetLoader):
    """Loads another loader's splits, keeping only the rows for which ``keep`` returns true."""

    def __init__(self, inner: DatasetLoader, keep: Callable[[dict[str, Any]], bool]) -> None:
        self._inner = inner
        self._keep = keep

    @override
    def load(self, name: str | None) -> DatasetDict:
        loaded = self._inner.load(name)
        return DatasetDict({split: data.filter(self._keep) for split, data in loaded.items()})

    @override
    def metadata(self) -> dict[str, str]:
        return self._inner.metadata()


@final
class Subset(DatasetPolicy):
    """Restricts another policy's dataset to the rows for which ``keep`` returns true, in every split.

    A benchmark whose items are a row-filtered subset of a larger dataset (e.g. GPQA's diamond subset,
    HLE's natively-multiple-choice subset) wraps the base policy in a ``Subset``. ``description`` names that
    subset for the rendered dataset docs; without it the docs only note that some rows are dropped.
    """

    def __init__(
        self, inner: DatasetPolicy, keep: Callable[[dict[str, Any]], bool], description: str | None = None
    ) -> None:
        self._inner = inner
        self._keep = keep
        self._description = description

    @override
    def loader(self, custom_hf_revision: str | None) -> DatasetLoader:
        return _SubsetLoader(self._inner.loader(custom_hf_revision), self._keep)

    @override
    def documentation(self) -> str:
        return f"{self._inner.documentation()}\n- Restricted to {self._description or 'a subset of its rows'}."


@final
class _SubjectColumnLoader(DatasetLoader):
    """Loads a fixed ``config`` (``None`` = the default config) from the inner loader, keeping only the rows
    whose ``column`` equals the requested subject (``name``)."""

    def __init__(self, inner: DatasetLoader, config: str | None, column: str) -> None:
        self._inner = inner
        self._config = config
        self._column = column

    @override
    def load(self, name: str | None) -> DatasetDict:
        loaded = self._inner.load(self._config)
        return DatasetDict(
            {split: data.filter(lambda row: row[self._column] == name) for split, data in loaded.items()}
        )

    @override
    def metadata(self) -> dict[str, str]:
        return self._inner.metadata()


@final
class SubjectColumn(DatasetPolicy):
    """Use this in case the raw dataset encodes the subject in a column, rather than the load
    configuration.
    """

    def __init__(self, inner: DatasetPolicy, config: str | None, column: str) -> None:
        self._inner = inner
        self._config = config
        self._column = column

    @override
    def loader(self, custom_hf_revision: str | None) -> DatasetLoader:
        return _SubjectColumnLoader(self._inner.loader(custom_hf_revision), self._config, self._column)

    @override
    def documentation(self) -> str:
        shared_config = f"`{self._config}`" if self._config is not None else "default"
        return (
            f"{self._inner.documentation()}\n"
            f"- Subjects share the {shared_config} config and are split by the `{self._column}` column."
        )
