import os
import random
from pathlib import Path
from typing import Any

import requests
from datasets import Dataset, DatasetDict, DownloadConfig, load_dataset
from huggingface_hub import HfApi
from huggingface_hub.errors import RevisionNotFoundError

from eval_framework.metrics.completion.accuracy_completion import AccuracyCompletion
from eval_framework.metrics.completion.f1 import F1
from eval_framework.tasks.base import NO_SUBJECT, RANDOM_SEED, BaseTask, Language, ResponseType, SubjectType


class SQUAD2(BaseTask[str]):
    """Squad v2 dataset: https://huggingface.co/datasets/rajpurkar/squad_v2"""

    NAME = "SQuAD2"
    DATASET_PATH = "rajpurkar/squad_v2"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [AccuracyCompletion, F1]
    SUBJECTS = [NO_SUBJECT]
    UNANSWERABLE_STR = "unanswerable"
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question", "Answer", "Context", "unanswerable"]
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        self.stop_sequences = [".\n"]
        self.max_tokens = 300  # the max length of the ground truth is 160 characters while the average is ~19
        self.rnd_choice_shuffle = random.Random()

    def _get_squad_urls(self) -> dict[str, str]:
        """Get the URLs for this SQUAD version."""
        return {
            "train": "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/train-v2.0.json",
            "validation": "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v2.0.json",
        }

    def _load_hf_dataset(self, **kwargs: Any) -> Any:
        """Load SQUAD dataset, falling back to JSON if HF fails."""
        # Validate HF revision if specified
        self._validate_hf_revision(kwargs.get("path", self.DATASET_PATH))

        # Try HuggingFace first
        try:
            return self._load_from_huggingface(**kwargs)
        except ValueError as e:
            if "Feature type 'List' not found" in str(e):
                import warnings

                warnings.warn(
                    f"Dataset {kwargs.get('path', self.DATASET_PATH)} has incompatible feature types "
                    "(List instead of Sequence), loading directly from JSON files"
                )
                return self._load_from_json(**kwargs)
            raise

    def _validate_hf_revision(self, dataset_path: str) -> None:
        """Validate HuggingFace revision if specified."""
        if self.HF_REVISION:
            try:
                HfApi().dataset_info(repo_id=dataset_path, revision=self.HF_REVISION, timeout=100.0)
            except RevisionNotFoundError:
                raise

    def _load_from_huggingface(self, **kwargs: Any) -> Any:
        """Load dataset from HuggingFace."""
        cache_dir = os.environ.get("HF_DATASET_CACHE_DIR", f"{Path.home()}/.cache/huggingface/datasets")
        download_config = DownloadConfig(cache_dir=cache_dir, max_retries=5)

        return load_dataset(
            **kwargs,
            revision=self.HF_REVISION,
            trust_remote_code=True,
            cache_dir=cache_dir,
            download_config=download_config,
        )

    def _load_from_json(self, **kwargs: Any) -> Dataset | DatasetDict:
        """Load SQUAD directly from GitHub JSON files."""
        urls = self._get_squad_urls()
        requested_split = kwargs.get("split")
        splits_to_load = [requested_split] if requested_split else list(urls.keys())

        datasets = {}
        for split in splits_to_load:
            if split not in urls:
                continue

            dataset = self._download_and_parse_split(split, urls[split])
            if dataset:
                datasets[split] = dataset

        if not datasets:
            raise ValueError(f"Failed to load any splits for {kwargs.get('path', self.DATASET_PATH)}")

        # Return single dataset or DatasetDict depending on what was requested
        return datasets[requested_split] if requested_split else DatasetDict(datasets)

    def _download_and_parse_split(self, split: str, url: str) -> Dataset | None:
        """Download and parse a single SQUAD split."""
        try:
            # Download the data
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            squad_data = response.json()

            # Flatten the nested structure
            examples = self._flatten_squad_data(squad_data)
            return Dataset.from_list(examples)

        except Exception as e:
            import warnings

            warnings.warn(f"Failed to download {split} split: {e}")
            return None

    def _flatten_squad_data(self, squad_data: dict) -> list[dict]:
        """Flatten nested SQUAD JSON structure into examples."""
        examples = []
        for article in squad_data["data"]:
            title = article["title"]
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    example = {
                        "id": qa["id"],
                        "title": title,
                        "context": context,
                        "question": qa["question"],
                        "answers": {
                            "text": [answer["text"] for answer in qa.get("answers", [])],
                            "answer_start": [answer["answer_start"] for answer in qa.get("answers", [])],
                        },
                    }

                    examples.append(example)
        return examples

    def _load_dataset(self, subject: SubjectType) -> None:
        name = subject if subject != NO_SUBJECT else None

        hf_dataset = self._load_hf_dataset(path=self.DATASET_PATH, name=name)
        self.dataset = {}

        self.rnd = random.Random(RANDOM_SEED)

        for split, data in hf_dataset.items():
            data_list = list(data)

            if split == self.SAMPLE_SPLIT:
                self.rnd.shuffle(data_list)

            if split in [self.SAMPLE_SPLIT, self.FEWSHOT_SPLIT]:
                self.dataset[split] = data_list

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        prompt = (
            "Given the following context, answer the question. If the question cannot be answered based "
            f"on the context alone, respond with '{self.UNANSWERABLE_STR}'.\n\n"
            "Context:\n"
            f"{item['context']}\n\n"
            f"Question:\n{item['question']}\nAnswer:"
        )
        return prompt

    def _get_ground_truth(self, item: dict[str, Any]) -> list[str]:
        text_ = item["answers"]["text"]
        ground_truth_for_unanswerable = [
            self.UNANSWERABLE_STR,
            self.UNANSWERABLE_STR + " ",
            self.UNANSWERABLE_STR.capitalize(),
        ]
        ground_truths = text_ if text_ else ground_truth_for_unanswerable
        return [f"{ground_truth}" for ground_truth in ground_truths]

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        target = self._get_ground_truth(item)[0]
        assert target is not None
        assert isinstance(target, str)
        return target


class SQUAD(SQUAD2):
    """Squad dataset: https://huggingface.co/datasets/rajpurkar/squad"""

    NAME = "SQuAD"
    DATASET_PATH = "rajpurkar/squad"

    def _get_squad_urls(self) -> dict[str, str]:
        """Override to provide SQUAD v1 URLs."""
        return {
            "train": "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/train-v1.1.json",
            "validation": "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json",
        }

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        prompt = (
            "Given the following context, answer the question.\n\n"
            "Context:\n"
            f"{item['context']}\n\n"
            f"Question:\n{item['question']}\n"
        )
        return prompt

    def _get_ground_truth(self, item: dict[str, Any]) -> list[str]:
        return item["answers"]["text"]
