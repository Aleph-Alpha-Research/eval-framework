"""Social IQa (English): https://huggingface.co/datasets/allenai/social_i_qa

Commonsense reasoning about social situations: a context and question with three answer options
(``answerA``/``answerB``/``answerC``) and a 1-indexed ``label`` marking the correct one. The registered
variant is OLMES-style: options are shown as space-prefixed lettered choices (" A. …") and the model is
scored over the letter labels.

``allenai/social_i_qa`` ships only a (no-longer-supported) loading script on its main branch, so it cannot
be loaded by a pinned commit under ``datasets`` >= 4. We load Hugging Face's auto-generated parquet branch
instead, falling back to the original AI2 Mosaic source if that branch is unavailable.
"""

import json
import os
import zipfile
from pathlib import Path
from typing import Any, final, override
from urllib.request import urlretrieve

from datasets import Dataset, DatasetDict, DownloadConfig, load_dataset

from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetLoader, DatasetPolicy
from eval_framework.tasks.task_style import MCStyle

SOCIAL_I_QA_DATASET_PATH = "allenai/social_i_qa"
# Hugging Face auto-generates a parquet copy of every dataset on this hidden branch; we load from it
# because the dataset itself only ships a loading script (unsupported since datasets 4.x) on its main branch.
_PARQUET_REVISION = "refs/convert/parquet"
_SOURCE_ZIP_URL = "https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip"
_ZIP_SUBDIR = "socialiqa-train-dev"


def _cache_dir() -> str:
    return os.environ.get("HF_DATASET_CACHE_DIR", f"{Path.home()}/.cache/huggingface/datasets")


def _load_from_ai2_source(cache_dir: str) -> DatasetDict:
    """Rebuild train/validation from AI2 Mosaic's original zip (jsonl rows paired with label files),
    used only if the Hugging Face parquet branch cannot be reached."""
    root = Path(cache_dir) / "social_i_qa_direct"
    extract_dir = root / _ZIP_SUBDIR
    if not extract_dir.exists() or not list(extract_dir.glob("*.jsonl")):
        zip_path = root / "socialiqa-train-dev.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        if not zip_path.exists():
            urlretrieve(_SOURCE_ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(zip_path.parent)

    def read_split(jsonl_path: Path, labels_path: Path) -> list[dict[str, Any]]:
        labels = labels_path.read_text(encoding="utf-8").splitlines()
        rows = []
        for idx, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines()):
            data = json.loads(line)
            rows.append(
                {
                    "context": data["context"],
                    "question": data["question"],
                    "answerA": data["answerA"],
                    "answerB": data["answerB"],
                    "answerC": data["answerC"],
                    "label": labels[idx].strip(),
                }
            )
        return rows

    return DatasetDict(
        {
            "train": Dataset.from_list(read_split(extract_dir / "train.jsonl", extract_dir / "train-labels.lst")),
            "validation": Dataset.from_list(read_split(extract_dir / "dev.jsonl", extract_dir / "dev-labels.lst")),
        }
    )


@final
class _SocialIqaLoader(DatasetLoader):
    """Loads social_i_qa from Hugging Face's parquet branch, falling back to the original AI2 source."""

    def __init__(self, revision: str) -> None:
        self._revision = revision

    @override
    def load(self, name: str | None) -> DatasetDict:
        cache_dir = _cache_dir()
        try:
            return load_dataset(
                SOCIAL_I_QA_DATASET_PATH,
                revision=self._revision,
                cache_dir=cache_dir,
                download_config=DownloadConfig(cache_dir=cache_dir, max_retries=5),
            )
        except Exception:
            return _load_from_ai2_source(cache_dir)

    @override
    def metadata(self) -> dict[str, str]:
        return {"dataset_path": SOCIAL_I_QA_DATASET_PATH}


@final
class _SocialIqaDataset(DatasetPolicy):
    """social_i_qa's data policy — see the module docstring for why it can't use a plain ``Pinned``."""

    @override
    def loader(self, custom_hf_revision: str | None) -> DatasetLoader:
        return _SocialIqaLoader(custom_hf_revision or _PARQUET_REVISION)

    @override
    def documentation(self) -> str:
        url = f"https://huggingface.co/datasets/{SOCIAL_I_QA_DATASET_PATH}"
        return (
            f"- Link to dataset: [{url}]({url})\n"
            "- This dataset ships only a loading script on its `main` branch, which newer `datasets` versions no "
            f"longer support; it is therefore loaded from Hugging Face's auto-generated parquet branch "
            f"`{_PARQUET_REVISION}` instead."
        )


@final
class SocialIqaReader(ChoiceReader):
    """Reads a Social IQa item: the shown question is the context followed by the question, with the correct
    option at the 1-indexed ``label``."""

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        return ChoiceFields(
            raw_question=f"{item['context']} {item['question']}".strip(),
            choices=[item["answerA"], item["answerB"], item["answerC"]],
            correct_index=int(item["label"]) - 1,
        )


def social_iqa_mc_olmes(dataset: DatasetPolicy | None = None) -> Benchmark:
    styler = MCStyle(question_prefix="Question: ", cue_text="Answer:", space_prefixed_labels=True)
    dataset_policy = dataset if dataset is not None else _SocialIqaDataset()
    return ComposedBenchmark.choice(
        id="SocialIQAMC_OLMES",
        reader=SocialIqaReader(),
        styler=styler,
        sample_split="train",
        fewshot_split="train",
        dataset_policy=dataset_policy,
        language=Language.ENG,
    )


SOCIAL_IQA_BENCHMARKS: list[Benchmark] = [social_iqa_mc_olmes()]
