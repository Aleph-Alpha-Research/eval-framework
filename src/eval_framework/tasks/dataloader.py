import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union

from datasets import Dataset, DatasetDict, DownloadConfig, Features, IterableDataset, IterableDatasetDict, load_dataset
from huggingface_hub import HfApi
from huggingface_hub.errors import RevisionNotFoundError


class Dataloader(ABC):
    @abstractmethod
    def set_features(self, features: Features) -> None:
        pass

    @abstractmethod
    def load(self, **kwargs: Any) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
        pass


class HFDataloader(Dataloader):
    def __init__(self) -> None:
        self.features = None

    def set_features(self, features: Features) -> None:
        self.features = features

    def load(self, **kwargs: Any) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
        # Check if the HF_REVISION is valid before loading the dataset
        if "revision" in kwargs:
            try:
                _ = HfApi().dataset_info(repo_id=kwargs["path"], revision=kwargs["hf_revision"], timeout=100.0)
            except Exception as e:
                if isinstance(e, RevisionNotFoundError):
                    raise e

        cache_dir = os.environ.get("HF_DATASET_CACHE_DIR", f"{Path.home()}/.cache/huggingface/datasets")
        download_config = DownloadConfig(cache_dir=cache_dir, max_retries=5)
        try:
            return load_dataset(
                **kwargs,
                trust_remote_code=True,
                cache_dir=cache_dir,
                download_config=download_config,
                features=self.features,
            )
        except Exception:
            return load_dataset(
                **kwargs,
                trust_remote_code=True,
                cache_dir=f"{Path.home()}/.cache/eval-framework",
                features=self.features,
            )
