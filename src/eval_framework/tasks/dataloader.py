import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from datasets import Dataset, DatasetDict, DownloadConfig, Features, IterableDataset, IterableDatasetDict, load_dataset
from huggingface_hub import HfApi
from huggingface_hub.errors import RevisionNotFoundError


class Dataloader(ABC):
    @abstractmethod
    def load(
        self, path: str, name: str | None = None, revision: str | None = None, features: Features | None = None
    ) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
        pass


class HFDataloader(Dataloader):
    def load(
        self, path: str, name: str | None = None, revision: str | None = None, features: Features | None = None
    ) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
        # Check if the HF_REVISION is valid before loading the dataset
        if revision:
            try:
                _ = HfApi().dataset_info(repo_id=path, revision=revision, timeout=100.0)
            except Exception as e:
                if isinstance(e, RevisionNotFoundError):
                    raise e

        cache_dir = os.environ.get("HF_DATASET_CACHE_DIR", f"{Path.home()}/.cache/huggingface/datasets")
        download_config = DownloadConfig(cache_dir=cache_dir, max_retries=5)
        try:
            return load_dataset(
                path=path,
                name=name,
                revision=revision,
                trust_remote_code=True,
                cache_dir=cache_dir,
                download_config=download_config,
                features=features,
            )
        except Exception:
            return load_dataset(
                name=name,
                path=path,
                revision=revision,
                trust_remote_code=True,
                cache_dir=f"{Path.home()}/.cache/eval-framework",
                features=features,
            )
