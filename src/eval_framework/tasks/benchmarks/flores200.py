import os
import random
from pathlib import Path
from typing import Any

import pycountry
from datasets import DownloadConfig, load_dataset
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

    def _load_hf_dataset(self, **kwargs: Any) -> Any:
        """Override to handle FLORES-200 encoding issues by using parquet files."""
        # Check if the HF_REVISION is valid before loading the dataset
        if self.HF_REVISION:
            try:
                _ = HfApi().dataset_info(repo_id=kwargs["path"], revision=self.HF_REVISION, timeout=100.0)
            except Exception as e:
                if isinstance(e, RevisionNotFoundError):
                    raise e

        cache_dir: str = os.environ.get("HF_DATASET_CACHE_DIR", f"{Path.home()}/.cache/huggingface/datasets")
        download_config = DownloadConfig(cache_dir=cache_dir, max_retries=5)

        # First, try to load using parquet files to bypass the problematic loading script
        try:
            # Try loading without the loading script by using data_files
            # This forces the dataset library to use the parquet files directly
            dataset = load_dataset(
                kwargs.get("path", self.DATASET_PATH),
                name=kwargs.get("name"),
                split=kwargs.get("split"),
                data_files=None,  # Let it auto-discover parquet files
                revision=self.HF_REVISION,
                # cache_dir=cache_dir,
                download_config=download_config,
            )

            return dataset

        except Exception:
            # If parquet loading fails, try the original method
            # Try the original loading with the loading script
            dataset = load_dataset(
                **kwargs,
                revision=self.HF_REVISION,
                cache_dir=cache_dir,
                download_config=download_config,
            )
            return dataset

    def _load_dataset(self, subject: SubjectType) -> None:
        # Store the subject (language pair) for use in other methods
        self.subject = subject

        # For FLORES, we need to load the dataset once with all languages
        # The subject (e.g., "eng_Latn-deu_Latn") determines which fields we use
        hf_dataset = self._load_hf_dataset(path=self.DATASET_PATH, name="all")
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
