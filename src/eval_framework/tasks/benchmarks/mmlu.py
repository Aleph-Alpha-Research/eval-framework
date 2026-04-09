import re
from typing import Any

from eval_framework.metrics.completion.accuracy_completion import AccuracyCompletion
from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.metrics.loglikelihood.bits_per_byte import BitsPerByteLoglikelihood
from eval_framework.metrics.loglikelihood.confidence_weighted_accuracy import ConfidenceWeightedAccuracy
from eval_framework.metrics.loglikelihood.dcs import DistributionalCorrectnessScore
from eval_framework.metrics.loglikelihood.ternary import TernaryScore
from eval_framework.tasks.base import BaseTask, Language, ResponseType, Sample
from eval_framework.tasks.task_style import BPBStyle, ClozeStyle, MCStyle
from eval_framework.tasks.utils import get_n_letters

MMLU_STEM = [
    "abstract_algebra",
    "astronomy",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "electrical_engineering",
    "elementary_mathematics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics",
    "machine_learning",
]

MMLU_HUMANITIES = [
    "formal_logic",
    "high_school_european_history",
    "high_school_us_history",
    "high_school_world_history",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "moral_disputes",
    "moral_scenarios",
    "philosophy",
    "prehistory",
    "professional_law",
    "world_religions",
]

MMLU_SOCIAL_SCIENCES = [
    "econometrics",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_microeconomics",
    "high_school_psychology",
    "human_sexuality",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
]

MMLU_OTHER = [
    "anatomy",
    "business_ethics",
    "clinical_knowledge",
    "college_medicine",
    "global_facts",
    "human_aging",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "nutrition",
    "professional_accounting",
    "professional_medicine",
    "virology",
]

MMLU_SUBJECTS = sorted(MMLU_STEM + MMLU_HUMANITIES + MMLU_SOCIAL_SCIENCES + MMLU_OTHER)


class MMLU(BaseTask[str]):
    """MMLU dataset: https://huggingface.co/datasets/cais/mmlu"""

    NAME = "MMLU"
    DATASET_PATH = "cais/mmlu"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "dev"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood, BitsPerByteLoglikelihood]
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


class MMLU_OLMES(MMLU):
    """
    MMLU with OLMES-style prompt: space before each label in the prompt (" A.", " B.", ...).
    """

    NAME = "MMLU_OLMES"

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        question = item["question"].strip()
        choices = "".join([f" {key}. {choice}\n" for key, choice in zip(self.keys, item["choices"])])
        return f"Question: {question}\n{choices}"


class FullTextMMLU(MMLU):
    """MMLU dataset but where the model is expected to replicate choice text, rather than just the key."""

    NAME = "Full Text MMLU"
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood, BitsPerByteLoglikelihood]
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


class MMLU_IDK(MMLU):
    NAME = "MMLU_IDK"
    METRICS = [
        AccuracyLoglikelihood,
        AccuracyNormLoglikelihood,
        ConfidenceWeightedAccuracy,
        DistributionalCorrectnessScore,
        TernaryScore,
    ]

    def _get_initial_prompt_text(self, item: dict[str, Any]) -> str:
        return (
            f"The following are multiple choice questions (with answers) about {item['subject']}. "
            "Answer only if you are confident, since mistakes may be penalised, while correct answers receive points. "
            "It is acceptable to answer with '?' if you are unsure, and you will receive 0 points."
        )

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        completions = super()._get_possible_completions(item)
        return (completions or []) + [" ?"]


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


class _MMLU_Base(BaseTask[str]):
    """Shared base for TASK_STYLER-based MMLU variants (Cloze, MC, BPB)."""

    DATASET_PATH = "cais/mmlu"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "dev"
    SUBJECTS = MMLU_SUBJECTS
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question", "Answer"] + get_n_letters(4)
    LANGUAGE = Language.ENG

    def _get_subject_name(self, item: dict[str, Any]) -> str:
        return " ".join(item["subject"].split("_"))

    def _get_initial_prompt_text(self, item: dict[str, Any]) -> str:
        return f"The following are multiple choice questions (with answers) about {self._get_subject_name(item)}:"

    def _get_raw_question(self, item: dict[str, Any]) -> str:
        return item["question"].strip()

    def _get_choices(self, item: dict[str, Any]) -> list[str]:
        return item["choices"]

    def _get_correct_index(self, item: dict[str, Any]) -> int:
        return item["answer"]


class MMLUCloze(_MMLU_Base):
    NAME = "MMLUCloze"
    TASK_STYLER = ClozeStyle()


class MMLUMC(_MMLU_Base):
    NAME = "MMLUMC"
    TASK_STYLER = MCStyle(space_prefixed_labels=True)


class MMLUBPB(_MMLU_Base):
    NAME = "MMLUBPB"
    TASK_STYLER = BPBStyle()


class MMLUOtherBPB(MMLUBPB):
    NAME = "MMLUOtherBPB"
    SUBJECTS = MMLU_OTHER


class MMLUStemBPB(MMLUBPB):
    NAME = "MMLUStemBPB"
    SUBJECTS = MMLU_STEM


class MMLUHumanitiesBPB(MMLUBPB):
    NAME = "MMLUHumanitiesBPB"
    SUBJECTS = MMLU_HUMANITIES


class MMLUSocialSciencesBPB(MMLUBPB):
    NAME = "MMLUSocialSciencesBPB"
    SUBJECTS = MMLU_SOCIAL_SCIENCES
