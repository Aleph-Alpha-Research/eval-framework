import os

from typing import Any
from pathlib import Path
from eval_framework.tasks.base import Language
from eval_framework.tasks.benchmarks.winogrande import WINOGRANDE

from datasets import DownloadConfig, load_dataset
from huggingface_hub import HfApi
from huggingface_hub.errors import RevisionNotFoundError

ANSWER_STR_TO_NUM = {"1": 0, "2": 1}


class WINOX(WINOGRANDE):
    """
    Wino-X is a parallel dataset of German, French, and Russian Winograd schemas, aligned with their English
    counterparts, used to examine whether neural machine translation models can perform coreference resolution that
    requires commonsense knowledge, and whether multilingual language models are capable of commonsense reasoning
    across multiple languages.

    Winogrande: https://arxiv.org/abs/1907.10641
    Wino-X: https://github.com/demelin/Wino-X
    Wino-X: https://huggingface.co/datasets/demelin/wino_x
    """

    DATASET_PATH = "demelin/wino_x"
    HF_REVISION = "7d82697fd52ac8b03e62aadfddc61077320f21e7"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "test"
    LANGUAGE_SHORT_CODE = ""

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        choices = self._extract_choices(item)
        # in winogrande answer is a string but in wino_x it is an int
        return f" {choices[ANSWER_STR_TO_NUM[str(item['answer'])]]}"

    def _extract_question(self, item: dict) -> str:
        question, _ = item[f"context_{self.LANGUAGE_SHORT_CODE}"].split("_")
        question = question.replace("  ", " ")
        return question.strip()

    def _extract_choices(self, item: dict) -> list[str]:
        _, choice_suffix = item[f"context_{self.LANGUAGE_SHORT_CODE}"].split("_")
        choice_suffix = choice_suffix.replace("  ", " ")
        choices = [
            choice + choice_suffix
            for choice in [item[f"option1_{self.LANGUAGE_SHORT_CODE}"], item[f"option2_{self.LANGUAGE_SHORT_CODE}"]]
        ]
        return choices

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


class WINOX_DE(WINOX):
    NAME = "WINOX_DE"
    SUBJECTS = ["lm_en_de"]
    LANGUAGE = Language.DEU
    LANGUAGE_SHORT_CODE = "de"


class WINOX_FR(WINOX):
    NAME = "WINOX_FR"
    SUBJECTS = ["lm_en_fr"]
    LANGUAGE = Language.FRA
    LANGUAGE_SHORT_CODE = "fr"
