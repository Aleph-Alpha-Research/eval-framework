import re
from typing import Any

from eval_framework.metrics.completion_metrics.accuracy_completion import AccuracyCompletion
from eval_framework.metrics.loglikelihood_metrics.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.tasks.base import BaseTask, Language, ResponseType, Sample
from eval_framework.tasks.utils import get_n_letters

MMLU_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


class MMLU(BaseTask[str]):
    """MMLU dataset: https://huggingface.co/datasets/cais/mmlu"""

    NAME = "MMLU"
    DATASET_PATH = "cais/mmlu"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "dev"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
    SUBJECTS = MMLU_SUBJECTS
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question", "Answer"] + get_n_letters(4)
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)

        self.keys = get_n_letters(4)

    def _get_subject_name(self, item: dict[str, Any]) -> str:
        return " ".join(item["subject"].split("_"))

    def _get_initial_prompt_text(self, item: dict[str, Any]) -> str:
        return f"The following are multiple choice questions (with answers) about {self._get_subject_name(item)}."

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        question = item["question"].strip()
        choices = "".join([f"{key}. {choice}\n" for key, choice in zip(self.keys, item["choices"])])
        return f"Question: {question}\n{choices}"

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        ground_truth = self._get_ground_truth(item)
        assert ground_truth is not None
        return f"{self._get_cue_text(item)}{ground_truth}"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        return f" {self.keys[item['answer']]}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [f" {key}" for key in self.keys]


class FullTextMMLU(MMLU):
    """MMLU dataset but where the model is expected to replicate choice text, rather than just the key."""

    NAME = "Full Text MMLU"
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question", "answers"] + get_n_letters(4)

    def _get_initial_prompt_text(self, item: dict[str, Any]) -> str:
        subject_name = self._get_subject_name(item)
        return f"""The following are multiple choice questions (with possible answers) about {subject_name}.
Answer with the full text of the correct answer."""

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        question = item["question"].strip()
        choices = "".join([f"- {choice}\n" for choice in item["choices"]])
        return f"Question: {question}\nPossible answers:\n{choices}"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        return f" {item['choices'][item['answer']]}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [f" {choice}" for choice in item["choices"]]


class MMLU_COT(MMLU):
    """
    MMLU dataset with instruction to summarize reasoning and conclude with answer.
    Inspired by https://arxiv.org/pdf/2411.15124 (Table 44)
    """

    NAME = "MMLU_COT"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [AccuracyCompletion]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question", "Therefore", "the", "answer", "is", "ANSWER_LETTER"] + get_n_letters(
        4
    )

    ANS_RE = re.compile(r"Therefore, the answer is: ([ABCD])")

    def __init__(self, num_fewshot: int = 0) -> None:
        assert num_fewshot == 0, "Fewshot is not supported for MMLU_COT"
        super().__init__(num_fewshot)
        self.stop_sequences: list[str] = ["Question:"]

    def _extract_answer(self, completion: str) -> str:
        match = self.ANS_RE.search(completion)
        if match:
            match_str = match.group(1)
            return match_str
        else:
            return "[invalid]"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return ""

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        return self.keys[item["answer"]]

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        for stop_sequence in self.stop_sequences:
            if stop_sequence in completion_text:
                completion_text = completion_text.split(stop_sequence)[0]
        return self._extract_answer(completion_text)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        question = item["question"].strip()
        choices = "\n".join([f"{key}. {choice}" for key, choice in zip(self.keys, item["choices"])])
        return f"Question: {question}\n{choices}"

    def _get_initial_prompt_text(self, item: dict[str, Any]) -> str:
        return (
            f"The following are multiple choice questions about {self._get_subject_name(item)}. "
            'Summarize your reasoning concisely, then conclude with "Therefore, the answer is: X", where X is '
            "one of A, B, C, or D."
        )
