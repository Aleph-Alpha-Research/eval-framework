import re
from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import AccuracyLoglikelihood, AccuracyNormLoglikelihood
from eval_framework.metrics.loglikelihood.bits_per_byte import BitsPerByteLoglikelihood
from eval_framework.metrics.completion.accuracy_completion import AccuracyCompletion
from eval_framework.tasks.base import BaseTask, Language, NO_SUBJECT, ResponseType


class JeopardyCompletion(BaseTask[str]):
    NAME = "JeopardyCompletion"
    DATASET_PATH = "soldni/jeopardy"
    SAMPLE_SPLIT = "train"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [AccuracyCompletion, BitsPerByteLoglikelihood]
    SUBJECTS = "mosaicml_gauntlet"
    PERTURBATION_UNMODIFIABLE_WORDS = ["Category", "Question", "Answer"]
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        self.max_tokens = 50
        self.stop_sequences: list[str] = ["\n\n", "Question:", "Category:"]

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        ctx = item.get("context", "")
        m = re.match(r"(.*?):\s*(.*)", ctx, re.DOTALL)
        category, question = m.group(1).strip(), m.group(2).strip()
        return f"Category: {category}\nQuestion: {question}\n"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        return item.get("continuation", "")


class JeopardyMC(BaseTask[str]):
    NAME = "JeopardyMC"
    DATASET_PATH = "allenai/jeopardy-gen2mc"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "test"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyCompletion]
    SUBJECTS = [NO_SUBJECT]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Category", "Question"]
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        ctx = item.get("context_original", "")
        m = re.match(r"(.*?):\s*(.*)", ctx, re.DOTALL)
        category, question = m.group(1).strip(), m.group(2).strip()
        texts = item.get("choices", {}).get("text", [])
        labels = item.get("choices", {}).get("label", [])
        if len(texts) != len(labels):
            raise ValueError(f"Number of choices ({len(texts)}) does not match number of labels ({len(labels)}). Please check the dataset.")
        if len(labels) != len(set(labels)):
            raise ValueError(f"Duplicate labels found in labels: {labels}. Please check the dataset.")
        options = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
        return f"Category: {category}\nQuestion: {question}\n{options}\n"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        return f" {item.get('answerKey')}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:  
        labels = item.get("choices", {}).get("label", [])
        return [f" {l}" for l in labels]


class JeopardyCloze(JeopardyMC):
    NAME = "JeopardyCloze"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyCompletion]

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        ctx = item.get("context_original", "")
        m = re.match(r"(.*?):\s*(.*)", ctx, re.DOTALL)
        category, question = m.group(1).strip(), m.group(2).strip()
        return f"Category: {category}\nQuestion: {question}\n"
    
    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:  
        texts = item.get("choices", {}).get("text", [])
        return [f" {t}" for t in texts]
    
    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        return f" {item.get('continuation_original')}"