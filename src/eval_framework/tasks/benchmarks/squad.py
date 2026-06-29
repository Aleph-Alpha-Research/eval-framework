import os
import random
from pathlib import Path
from typing import Any

import requests
from datasets import Dataset, DatasetDict, DownloadConfig, load_dataset
from huggingface_hub import HfApi
from huggingface_hub.errors import RevisionNotFoundError

from eval_framework.metrics.completion.accuracy_completion import AccuracyCompletion
from eval_framework.metrics.completion.f1 import F1, F1SquadNormalized
from eval_framework.metrics.loglikelihood.bits_per_byte import BitsPerByteLoglikelihood
from eval_framework.tasks.base import NO_SUBJECT, RANDOM_SEED, BaseTask, Language, ResponseType, SubjectType
from eval_framework.tasks.dataset_revisions import HF_REVISIONS_LOCKFILE


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
        if self.hf_revision:
            try:
                HfApi().dataset_info(repo_id=dataset_path, revision=self.hf_revision, timeout=100.0)
            except RevisionNotFoundError:
                raise

    def _load_from_huggingface(self, **kwargs: Any) -> Any:
        """Load dataset from HuggingFace."""
        cache_dir = os.environ.get("HF_DATASET_CACHE_DIR", f"{Path.home()}/.cache/huggingface/datasets")
        download_config = DownloadConfig(cache_dir=cache_dir, max_retries=5)

        return load_dataset(
            **kwargs,
            revision=self.hf_revision,
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
        return [f" {ground_truth}" for ground_truth in ground_truths]

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        target = self._get_ground_truth(item)[0]
        assert target is not None
        assert isinstance(target, str)
        return target


class SQUAD2BPB(SQUAD2):
    """
    SQuAD2 variant that scores loglikelihood of the gold answer text.
    Reports bits-per-byte on the reference answer (first acceptable answer).
    """

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "SQuAD2 BPB"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [BitsPerByteLoglikelihood]

    def _get_ground_truth(self, item: dict[str, Any]) -> list[str]:
        text_ = item["answers"]["text"]
        if text_:
            return [f" {text_[0]}"]
        return [f" {self.UNANSWERABLE_STR}"]

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return self._get_ground_truth(item)

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        gt_list = self._get_ground_truth(item)
        assert gt_list
        return gt_list[0]


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


class SQuAD2_MA(SQUAD2):
    """SQuAD v2 with the exact system prompt used in MA training"""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "SQuAD2_MA"
    UNANSWERABLE_STR = "unanswerable"
    # Merlin-Arthur RAG-specific instruction. The context shown to the model may be
    # partially masked (during training, masks are produced by probing the policy),
    # so this tells the model how to treat hidden spans. It lives in the serving
    # system prompt so masking probes and rollouts share identical conditions.
    MASKED_RAG_PROMPT = (
        "Parts of the context may be hidden and replaced with '...'. Base your answer "
        "only on the information that remains visible; do not guess at hidden content."
    )

    SYSTEM_PROMPT = (
        "You are given a context and a question. Answer the question based ONLY on the "
        "information provided in the context. If the context does not contain enough "
        "information to answer the question, say 'I don't know'.\n\n"
        f"{MASKED_RAG_PROMPT}\n\n"
        "Think step by step inside <think>...</think> tags, then provide your final answer "
        f"after '{UNANSWERABLE_STR}'.\n\n"
        "Format your response as:\n"
        "<think>\n[your reasoning]\n</think>\n"
        f"{UNANSWERABLE_STR} [your answer]"
    )

    METRICS = [AccuracyCompletion, F1, F1SquadNormalized]

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        self.stop_sequences = []
        self.max_tokens = 30_000


    def _get_system_prompt_text(self, item: dict[str, Any]) -> str | None:
        return self.SYSTEM_PROMPT

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"Context:\n{item['context']}\n\nQuestion:\n{item['question']}\n"

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        """Clean up the generated answer."""
        # Remove common prefixes and clean whitespace
        cleaned = completion_text.strip()
        if cleaned.startswith("Answer:"):
            cleaned = cleaned[7:].strip()
        return cleaned

    def _get_ground_truth(self, item: dict[str, Any]) -> list[str]:
        text_ = item["answers"]["text"]
        ground_truth_for_unanswerable = [
            self.UNANSWERABLE_STR,
            self.UNANSWERABLE_STR + " ",
            self.UNANSWERABLE_STR.capitalize(),
        ]
        ground_truths = text_ if text_ else ground_truth_for_unanswerable
        return ground_truths

class SQuAD2_MA_NO_SYSPROMT(SQuAD2_MA):
    SYSTEM_PROMPT = ""

class SQuAD_OLMES(SQUAD):
    """SQuAD variant matching OLMES implementation."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "SQuAD_OLMES"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "train"
    PERTURBATION_UNMODIFIABLE_WORDS = ["Title", "Background", "Question", "Answer"]
    METRICS = [F1SquadNormalized]

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        self.stop_sequences = ["Title:", "\n\n"]
        self.max_tokens = 50

    def _get_initial_prompt_text(self, item: dict[str, Any]) -> str:
        return (
            "The following are reading comprehension questions, "
            "where the answer to each question is a segment of text from the corresponding background text."
        )

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        # For fewshot, we only need to return the first ground truth.
        target = self._get_ground_truth(item)[0]
        assert target is not None
        assert isinstance(target, str)
        return f"{self._get_cue_text(item)}{target}"

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"Title: {item['title']}\nBackground: {item['context']}\nQuestion: {item['question']}\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> list[str]:
        # SQuAD might have multiple ground truths, so we return a list.
        return [f" {a}" for a in item["answers"]["text"]]
