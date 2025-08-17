import logging
import os
import urllib.request
from typing import Any

from datasets import load_dataset

from eval_framework.tasks.benchmarks.hellaswag import HELLASWAG

logger = logging.getLogger(__name__)


class HELLASWAG_LOCAL(HELLASWAG):
    """
    Identical to the HELLASWAG task, but loads data from a local source instead of downloading it.
    This class demonstrates how to implement a task that utilizes locally stored datasets.
    Data source: https://github.com/rowanz/hellaswag
    """

    NAME = "HellaSwag_local"

    def download_data_from_url(self, url: str) -> None:
        """Download a file from a URL and save it locally."""
        target_filename = url.split("/")[-1] if "/" in url else url
        target_filepath = os.path.join(".data", target_filename)
        if not os.path.exists(".data"):
            os.makedirs(".data")
        if not os.path.exists(target_filepath):
            logger.info(f"Downloading {target_filename} from {url}...")
            urllib.request.urlretrieve(url, ".data/" + target_filename)

    def _load_hf_dataset(self, **kwargs: Any) -> Any:
        """Create a Hugging Face dataset from local JSONL files.
        https://huggingface.co/docs/datasets/v1.1.3/loading_datasets.html#json-files
        """

        self.download_data_from_url(
            "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl"
        )
        self.download_data_from_url(
            "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
        )

        return load_dataset(
            "json",
            data_files={
                "train": ".data/hellaswag_train.jsonl",
                "validation": ".data/hellaswag_val.jsonl",
            },
        )
