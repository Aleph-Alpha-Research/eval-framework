"""
Social IQA: Commonsense reasoning about social interactions.

Reference: oe_eval/tasks/oe_eval_tasks/siqa.py
Dataset: allenai/social_i_qa (context, question, answerA/B/C, label 1-indexed).
"""

from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.tasks.base import NO_SUBJECT, BaseTask, Language, ResponseType
from eval_framework.tasks.utils import get_n_letters


def _social_iqa_context_question(item: dict[str, Any]) -> str:
    """Match oe_eval: context + " " + question (used for both cloze and MC)."""
    context = item.get("context", "")
    question = item.get("question", "")
    return f"{context} {question}".strip()


class SocialIQACloze(BaseTask[str]):
    """
    Social IQA cloze: loglikelihood over full answer text.

    Prompt matches oe_eval make_cloze_prompt: "Question: {context} {question}\\nAnswer:"
    """

    NAME = "SocialIQACloze"
    DATASET_PATH = "allenai/social_i_qa"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
    SUBJECTS = [NO_SUBJECT]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question"]
    LANGUAGE = Language.ENG

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

    Prompt matches oe_eval make_mcq_prompt: Question + choices with " A.", " B.", " C.".
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
