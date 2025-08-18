import random
import re
from abc import ABC
from typing import Any

from eval_framework.metrics.base import BaseMetric
from eval_framework.metrics.completion_metrics.accuracy_completion import AccuracyCompletion
from eval_framework.tasks.base import RANDOM_SEED, BaseTask, Language, ResponseType, Sample


class DUC(BaseTask[str], ABC):
    """https://huggingface.co/datasets/midas/duc2001"""

    DATASET_PATH: str = "midas/duc2001"
    SAMPLE_SPLIT: str = "test"
    FEWSHOT_SPLIT: str = "test"
    RESPONSE_TYPE: ResponseType = ResponseType.COMPLETION
    METRICS: list[type[BaseMetric]] = [AccuracyCompletion]
    SUBJECTS: list[str] = ["raw"]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Text", "Keyphrase"]
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)

        self.stop_sequences: list[str] = ["Text:"]
        self.max_tokens = 50  # longest keyphrase is less than 50 characters long

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        for stop_sequence in self.stop_sequences:
            if stop_sequence in completion_text:
                completion_text = completion_text.split(stop_sequence)[0]
        completion_text = completion_text.strip()
        return completion_text

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        instruction_text = " ".join(item["document"])
        instruction_text = re.sub(r"\s+([.,!?;:])", r"\1", instruction_text)
        return f"Text: {instruction_text}\nKeyphrase:"

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        target = self._get_ground_truth(item)
        assert target is not None
        assert isinstance(target, list)
        return f" {target[0]}"


class DUC_EXTRACTIVE(DUC):
    NAME = "DUC Extractive"
    SUBJECTS: list[str] = ["raw"]

    def _get_ground_truth(self, item: dict[str, Any]) -> list[str]:
        return item["extractive_keyphrases"]

    def _get_system_prompt_text(self, item: dict[str, Any]) -> str:
        return (
            "You are an AI model tasked with extracting keyphrases from a text document. "
            "Keyphrases should capture main ideas or significant topics exactly as worded in the text."
        )


class DUC_ABSTRACTIVE(DUC):
    NAME = "DUC Abstractive"
    SUBJECTS: list[str] = ["raw"]

    def _get_ground_truth(self, item: dict[str, Any]) -> list[str]:
        return item["abstractive_keyphrases"]

    def _load_dataset(self, subject: str) -> None:
        # not all samples have abstractive keyphrases
        hf_dataset = self._load_hf_dataset(path=self.DATASET_PATH, name=subject)
        self.dataset = {}

        for split, data in hf_dataset.items():
            data_list = list(filter(lambda x: len(x["abstractive_keyphrases"]) > 0, data))

            if split == self.SAMPLE_SPLIT:
                self.rnd = random.Random(RANDOM_SEED)
                self.rnd.shuffle(data_list)

            if split in [self.SAMPLE_SPLIT, self.FEWSHOT_SPLIT]:
                self.dataset[split] = data_list

    def _get_system_prompt_text(self, item: dict[str, Any]) -> str:
        return (
            "You are an AI model tasked with generating abstractive keyphrases "
            "that capture the main ideas of the text without using exact wording."
        )

    def _get_initial_prompt_text(self, item: dict[str, Any]) -> str:
        return "Paraphrase the following texts to improve clarity and relevance."
