import re
from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.metrics.loglikelihood.bits_per_byte import BitsPerByteLoglikelihood
from eval_framework.metrics.loglikelihood.confidence_weighted_accuracy import ConfidenceWeightedAccuracy
from eval_framework.metrics.loglikelihood.dcs import DistributionalCorrectnessScore
from eval_framework.metrics.loglikelihood.ternary import TernaryScore
from eval_framework.tasks.base import NO_SUBJECT, BaseTask, Language, ResponseType
from eval_framework.tasks.task_style import BPBStyle, ClozeStyle, MCStyle


class HELLASWAG(BaseTask[str]):
    """Hellaswag dataset: https://huggingface.co/datasets/Rowan/hellaswag
    available data set sections: train, validation, test"""

    NAME = "HellaSwag"
    DATASET_PATH = "Rowan/hellaswag"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood, BitsPerByteLoglikelihood]
    SUBJECTS = [NO_SUBJECT]
    LANGUAGE = Language.ENG

    @staticmethod
    def _preprocess(prompt: str) -> str:
        # remove bracketed text
        prompt = prompt.strip()
        prompt = prompt.replace(" [title]", ". ")
        prompt = re.sub("\\[.*?\\]", "", prompt)
        prompt = prompt.replace("  ", " ")
        return prompt

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        subject = self._preprocess(item["activity_label"])
        question = self._preprocess(item["ctx_a"] + " " + item["ctx_b"].capitalize()).strip()
        return f"{subject}: {question}"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        ground_truth_index = int(item["label"] if item["label"] != "" else 0)
        choices = [self._preprocess(ending) for ending in item["endings"]]
        return f" {choices[ground_truth_index]}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [f" {self._preprocess(ending)}" for ending in item["endings"]]


class HELLASWAG_OLMES(HELLASWAG):
    NAME = "HellaSwag_OLMES"
    SAMPLE_SPLIT = "train"


class HELLASWAG_IDK(HELLASWAG):
    NAME = "HellaSwag_IDK"
    METRICS = [
        AccuracyLoglikelihood,
        AccuracyNormLoglikelihood,
        ConfidenceWeightedAccuracy,
        DistributionalCorrectnessScore,
        TernaryScore,
    ]

    def _get_initial_prompt_text(self, item: dict[str, Any]) -> str:
        return (
            "Complete the sentence only if you are confident, since mistakes may be penalised, while correct "
            "completions receive points. It is acceptable to answer with 'I do not know' if you are unsure, "
            "and you will receive 0 points."
        )

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        completions = super()._get_possible_completions(item)
        return (completions or []) + [" I do not know."]


class _HELLASWAG_Base(BaseTask[str]):
    """Shared base for HELLASWAG variants (Cloze, MC, BPB).

    Subclasses set ``NAME`` and ``TASK_STYLER``; everything else is inherited.
    """

    DATASET_PATH = "Rowan/hellaswag"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "train"
    SUBJECTS = [NO_SUBJECT]
    LANGUAGE = Language.ENG

    @staticmethod
    def _preprocess(prompt: str) -> str:
        # remove bracketed text
        prompt = prompt.strip()
        prompt = prompt.replace(" [title]", ". ")
        prompt = re.sub("\\[.*?\\]", "", prompt)
        prompt = prompt.replace("  ", " ")
        prompt = re.sub(r"\.\. ", ". ", prompt)
        return prompt

    def _get_choices(self, item: dict[str, Any]) -> list[str]:
        return [self._preprocess(ending) for ending in item["endings"]]

    def _get_raw_question(self, item: dict[str, Any]) -> str:
        # Include activity_label as prefix to match the OLMES prompt format:
        # "ActivityLabel: preprocessed_context"
        subject = self._preprocess(item["activity_label"])
        context = self._preprocess(item["ctx_a"] + " " + item["ctx_b"].capitalize()).strip()
        return f"{subject}: {context}"

    def _get_correct_index(self, item: dict[str, Any]) -> int:
        return int(item["label"] if item["label"] != "" else 0)


class HELLASWAGCloze(_HELLASWAG_Base):
    NAME = "HELLASWAGCloze"
    TASK_STYLER = ClozeStyle()


class HELLASWAGMC(_HELLASWAG_Base):
    NAME = "HELLASWAGMC"
    TASK_STYLER = MCStyle(space_prefixed_labels=True)


class HELLASWAGBPB(_HELLASWAG_Base):
    NAME = "HellaSwagBPB"
    TASK_STYLER = BPBStyle(question_prefix="", cue_text="", trailing_newline=False)
