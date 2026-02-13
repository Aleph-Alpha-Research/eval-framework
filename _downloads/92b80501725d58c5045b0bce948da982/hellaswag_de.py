import re
from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.tasks.base import NO_SUBJECT, BaseTask, Language, ResponseType


class HELLASWAG_DE(BaseTask[str]):
    """Hellaswag dataset: https://huggingface.co/datasets/LeoLM/HellaSwag_de
    available data set sections: train (1k rows), validation (10k rows)"""

    NAME = "HellaSwag German"
    DATASET_PATH = "LeoLM/HellaSwag_de"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
    SUBJECTS = [NO_SUBJECT]
    LANGUAGE = Language.DEU

    @staticmethod
    def _preprocess(prompt: str) -> str:
        # remove bracketed text
        prompt = prompt.strip()
        prompt = prompt.replace(" [title]", ". ")
        prompt = re.sub("\\[.*?\\]", "", prompt)
        prompt = prompt.replace("  ", " ")
        return prompt

    def _load_dataset(self, subject: str) -> None:
        super()._load_dataset(subject)
        new_dataset = {}
        for split, items in self.dataset.items():
            # in the valid split, only 10035 out of 10042 items are well translated
            new_dataset[split] = [item for item in items if len(item["endings_de"]) == len(item["endings"])]
        self.dataset = new_dataset

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        subject = self._preprocess(item["activity_label_de"])
        question = self._preprocess(item["ctx_de"]).strip()
        return f"{subject}: {question}"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        ground_truth_index = int(item["label"] if item["label"] != "" else 0)
        choices = [self._preprocess(ending) for ending in item["endings_de"]]
        return f" {choices[ground_truth_index]}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [f" {self._preprocess(ending)}" for ending in item["endings_de"]]
