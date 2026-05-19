import os
import random
from pathlib import Path
from typing import Any

import pycountry
from datasets import DatasetDict, DownloadConfig, load_dataset
from huggingface_hub import HfApi
from huggingface_hub.errors import RevisionNotFoundError

from eval_framework.metrics.completion.bleu import BLEU
from eval_framework.tasks.base import RANDOM_SEED, BaseTask, Language, ResponseType, Sample, SubjectType

FLORES_LANGUAGES = [
    "deu_Latn",
    "eng_Latn",
    "fin_Latn",
    "fra_Latn",
    "nld_Latn",
]  # Note: there are many more languages in the dataset, but we only consider these for now


class Flores200(BaseTask[str]):
    """FLORES-200 dataset: https://huggingface.co/datasets/facebook/flores"""

    NAME = "FLoRes-200"
    DATASET_PATH = "facebook/flores"
    HF_REVISION = "fd7d8f42fccb9dbc35830053a8c705a2627124ce"
    SAMPLE_SPLIT = "devtest"
    FEWSHOT_SPLIT = "dev"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [BLEU]
    SUBJECTS = [f"{s}-{t}" for s in FLORES_LANGUAGES for t in FLORES_LANGUAGES if s != t]
    PERTURBATION_UNMODIFIABLE_WORDS = ["sentence"]
    LANGUAGE = {
        "deu_Latn": Language.DEU,
        "eng_Latn": Language.ENG,
        "fin_Latn": Language.FIN,
        "fra_Latn": Language.FRA,
        "nld_Latn": Language.NLD,
    }

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        self.stop_sequences = ["\n"]

    def _load_hf_dataset_for_subject(self, subject: SubjectType) -> DatasetDict:
        """Load FLORES-200 parquet files for a specific language pair.

        The datasets library removed supports for loading scripts, so we load
        parquet files directly via hf:// URIs pinned to the specific revision.
        """
        # Check if the HF_REVISION is valid before loading the dataset
        if self.HF_REVISION:
            try:
                _ = HfApi().dataset_info(repo_id=self.DATASET_PATH, revision=self.HF_REVISION, timeout=100.0)
            except Exception as e:
                if isinstance(e, RevisionNotFoundError):
                    raise e

        cache_dir: str = os.environ.get("HF_DATASET_CACHE_DIR", f"{Path.home()}/.cache/huggingface/datasets")
        download_config = DownloadConfig(cache_dir=cache_dir, max_retries=5)

        # Reference for loading parquet files: https://huggingface.co/docs/datasets/en/loading#parquet
        base_uri = f"https://huggingface.co/datasets/{self.DATASET_PATH}/resolve/{self.HF_REVISION}/{subject}"
        data_files = {
            self.FEWSHOT_SPLIT: f"{base_uri}/{self.FEWSHOT_SPLIT}.parquet",
            self.SAMPLE_SPLIT: f"{base_uri}/{self.SAMPLE_SPLIT}.parquet",
        }

        return load_dataset(
            "parquet",
            data_files=data_files,
            cache_dir=cache_dir,
            download_config=download_config,
        )

    def _load_dataset(self, subject: SubjectType) -> None:
        # Store the subject (language pair) for use in other methods
        self.subject = subject

        # Load parquet files for each subject
        hf_dataset = self._load_hf_dataset_for_subject(subject)
        self.dataset = {}

        self.rnd = random.Random(RANDOM_SEED)

        for split, data in hf_dataset.items():
            data_list = list(data)

            # Add the subject to each item so _get_instruction_text can use it
            for item in data_list:
                item["subject"] = subject

            if split == self.SAMPLE_SPLIT:
                self.rnd.shuffle(data_list)
                self.dataset[split] = data_list
            elif split == self.FEWSHOT_SPLIT:
                self.dataset[split] = data_list

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        source_key = item["subject"].split("-")[0]
        source_language = pycountry.languages.get(alpha_3=source_key.split("_")[0]).name
        source = item[f"sentence_{source_key}"]
        instruction = f"{source_language} sentence: {source}\n"
        target_key = item["subject"].split("-")[1]
        target_language = pycountry.languages.get(alpha_3=target_key.split("_")[0]).name

        return f"{instruction}{target_language} sentence:"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        target_key = item["subject"].split("-")[1]
        return item[f"sentence_{target_key}"]

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        target = f" {self._get_ground_truth(item)}"
        assert target is not None
        assert isinstance(target, str)
        return target

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        return completion_text.strip()
