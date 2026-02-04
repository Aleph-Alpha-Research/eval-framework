from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.metrics.loglikelihood.bits_per_byte import BitsPerByteLoglikelihood
from eval_framework.metrics.completion.accuracy_completion import AccuracyCompletion
from eval_framework.tasks.base import NO_SUBJECT, BaseTask, Language, ResponseType
from eval_framework.tasks.utils import get_n_letters

# TO-DO: This eval dataset requires remote code execution to load, and it would be ideal to load
# a version which does not require this for security reasons.

class BasicSkillsCloze(BaseTask[str]):
    """Basic arithmetic skills benchmark using allenai/basic-skills."""

    NAME = "BasicSkillsCloze"
    DATASET_PATH = "allenai/basic-skills"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "validation"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood, BitsPerByteLoglikelihood]
    SUBJECTS = [NO_SUBJECT]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question", "Answer"]
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        self.keys = get_n_letters(10)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"Question: {item['question']}\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        return f" {item['answer']}"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        choices = [*item["wrong_answers"], item["answer"]]
        return [f" {choice}" for choice in choices]


class BasicSkillsMC(BasicSkillsCloze):
    """
    Multiple-choice variant of BasicSkills where the model selects a letter (A, B, C, ...).
    """

    NAME = "BasicSkillsMC"

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        question = item["question"]
        choices = [*item["wrong_answers"], item["answer"]]
        labels = self.keys[: len(choices)]
        options = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices))
        return f"Question: {question}\n{options}\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        choices = [*item["wrong_answers"], item["answer"]]
        correct_index = len(choices) - 1
        return f" {self.keys[correct_index]}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        num_choices = len(item["wrong_answers"]) + 1
        return [f" {key}" for key in self.keys[:num_choices]]

