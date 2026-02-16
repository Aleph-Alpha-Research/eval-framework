"""
Social IQA: Commonsense reasoning about social interactions.

Dataset: allenai/social_i_qa (context, question, answerA/B/C, label 1-indexed).
"""

import json
import os
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

from datasets import Dataset, DatasetDict, DownloadConfig, load_dataset

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.tasks.base import NO_SUBJECT, BaseTask, Language, ResponseType
from eval_framework.tasks.utils import get_n_letters

SOCIAL_I_QA_DATASET_PATH = "allenai/social_i_qa"
SOCIAL_I_QA_PARQUET_REVISION = "refs/convert/parquet"
SOCIAL_I_QA_SOURCE_URL = "https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip"
SOCIAL_I_QA_ZIP_SUBDIR = "socialiqa-train-dev"


def _load_social_i_qa_parquet() -> DatasetDict:
    """Load social_i_qa from parquet (Hub parquet branch or explicit parquet URLs).

    Load parquet files directly so we do not depend on dataset loading scripts, which
    are no longer supported in datasets 4.x.
    """
    cache_dir: str = os.environ.get("HF_DATASET_CACHE_DIR", f"{Path.home()}/.cache/huggingface/datasets")
    download_config = DownloadConfig(cache_dir=cache_dir, max_retries=5)

    try:
        return load_dataset(
            SOCIAL_I_QA_DATASET_PATH,
            revision=SOCIAL_I_QA_PARQUET_REVISION,
            cache_dir=cache_dir,
            download_config=download_config,
        )
    except Exception:
        pass

    base_uri = (
        f"https://huggingface.co/datasets/{SOCIAL_I_QA_DATASET_PATH}/resolve/"
        f"{SOCIAL_I_QA_PARQUET_REVISION.replace('/', '%2F')}/default"
    )
    data_files = {
        "train": f"{base_uri}/train-00000-of-00001.parquet",
        "validation": f"{base_uri}/validation-00000-of-00001.parquet",
    }
    return load_dataset(
        "parquet",
        data_files=data_files,
        cache_dir=cache_dir,
        download_config=download_config,
    )


def _load_social_i_qa_direct() -> DatasetDict:
    """Load social_i_qa by downloading and processing the original source (AI2 Mosaic).

    Third fallback when trust_remote_code and parquet are unavailable. Replicates the
    logic from the dataset loading script: download zip, read jsonl + label files,
    and build train/validation splits with the expected schema.
    """
    cache_dir = Path(os.environ.get("HF_DATASET_CACHE_DIR", f"{Path.home()}/.cache/huggingface/datasets"))
    extract_dir = cache_dir / "social_i_qa_direct" / SOCIAL_I_QA_ZIP_SUBDIR
    zip_path = cache_dir / "social_i_qa_direct" / "socialiqa-train-dev.zip"

    if not extract_dir.exists() or not list(extract_dir.glob("*.jsonl")):
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        if not zip_path.exists():
            urlretrieve(SOCIAL_I_QA_SOURCE_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(zip_path.parent)

    def _read_split(jsonl_path: Path, label_path: Path) -> list[dict[str, Any]]:
        with open(label_path, encoding="utf-8") as f:
            labels = [line.strip() for line in f]
        rows = []
        with open(jsonl_path, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                rows.append(
                    {
                        "context": data["context"],
                        "question": data["question"],
                        "answerA": data["answerA"],
                        "answerB": data["answerB"],
                        "answerC": data["answerC"],
                        "label": labels[idx],
                    }
                )
        return rows

    train_data = _read_split(
        extract_dir / "train.jsonl",
        extract_dir / "train-labels.lst",
    )
    validation_data = _read_split(
        extract_dir / "dev.jsonl",
        extract_dir / "dev-labels.lst",
    )

    return DatasetDict(
        {
            "train": Dataset.from_list(train_data),
            "validation": Dataset.from_list(validation_data),
        }
    )


def _load_social_i_qa() -> DatasetDict:
    """Load social_i_qa using the Hugging Face dataset API, with fallbacks."""
    try:
        return load_dataset(
            SOCIAL_I_QA_DATASET_PATH,
            trust_remote_code=True,
        )
    except (RuntimeError, TypeError) as e:
        if "no longer supported" in str(e) or "trust_remote_code" in str(e).lower():
            try:
                return _load_social_i_qa_parquet()
            except Exception:
                try:
                    return _load_social_i_qa_direct()
                except Exception as direct_err:
                    raise RuntimeError(
                        "allenai/social_i_qa could not be loaded: dataset scripts are no longer "
                        "supported, parquet fallback failed, and direct download from source failed. "
                        "Pin datasets<4.0 or check network access to the dataset source."
                    ) from direct_err
        raise


def _social_iqa_context_question(item: dict[str, Any]) -> str:
    context = item.get("context", "")
    question = item.get("question", "")
    return f"{context} {question}".strip()


class SocialIQACloze(BaseTask[str]):
    """
    Social IQA cloze: loglikelihood over full answer text.
    """

    NAME = "SocialIQACloze"
    DATASET_PATH = SOCIAL_I_QA_DATASET_PATH
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
    SUBJECTS = [NO_SUBJECT]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question"]
    LANGUAGE = Language.ENG

    def _load_dataset(self, subject: Any) -> None:
        hf_dataset = _load_social_i_qa()
        self.dataset = self._shuffle_splits(hf_dataset=hf_dataset)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        query = _social_iqa_context_question(item)
        return f"Question: {query}\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        # label is 1-indexed (1, 2, 3) in the dataset
        idx = int(item["label"]) - 1
        choices = [item["answerA"], item["answerB"], item["answerC"]]
        return f" {choices[idx]}"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        choices = [item["answerA"], item["answerB"], item["answerC"]]
        return [f" {c}" for c in choices]


class SocialIQAMC(SocialIQACloze):
    """
    Social IQA multiple choice: loglikelihood over A/B/C.
    """

    NAME = "SocialIQAMC"

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        query = _social_iqa_context_question(item)
        choices = [item["answerA"], item["answerB"], item["answerC"]]
        labels = get_n_letters(len(choices))
        # Match oe_eval label_format " A." for tokenization
        options = "\n".join(f" {label}. {choice}" for label, choice in zip(labels, choices))
        return f"Question: {query}\n{options}\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        idx = int(item["label"]) - 1
        labels = get_n_letters(3)
        return f" {labels[idx]}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [" A", " B", " C"]
