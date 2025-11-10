import random
import re
from typing import Any

from eval_framework.metrics.completion.accuracy_completion import AccuracyCompletion
from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.metrics.loglikelihood.confidence_weighted_accuracy import ConfidenceWeightedAccuracy
from eval_framework.metrics.loglikelihood.distributional_correctness_score import DistributionalCorrectnessScore
from eval_framework.metrics.loglikelihood.ternary import TernaryScore
from eval_framework.tasks.base import NO_SUBJECT, RANDOM_SEED, BaseTask, Language, ResponseType, Sample
from eval_framework.tasks.utils import get_n_letters

MMLU_PRO_SUBJECTS = [
    "engineering",
    "physics",
    "psychology",
    "chemistry",
    "biology",
    "law",
    "philosophy",
    "computer science",
    "other",
    "economics",
    "business",
    "history",
    "math",
    "health",
]


class MMLU_PRO(BaseTask[str]):
    """MMLU_PRO dataset: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro"""

    NAME = "MMLU Pro"
    DATASET_PATH = "TIGER-Lab/MMLU-Pro"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "test"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
    SUBJECTS = MMLU_PRO_SUBJECTS
    PERTURBATION_UNMODIFIABLE_WORDS = get_n_letters(10)
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)

        self.keys = get_n_letters(10)

    def _load_dataset(self, subject: str) -> None:
        name = subject if subject != NO_SUBJECT else None

        hf_dataset = self._load_hf_dataset(path=self.DATASET_PATH)

        hf_dataset = hf_dataset.filter(lambda example: example["category"] == name)

        self.dataset = {}
        for split, data in hf_dataset.items():
            data_list = list(data)
            assert len(data_list) > 0

            if split == self.SAMPLE_SPLIT:
                self.rnd = random.Random(RANDOM_SEED)
                self.rnd.shuffle(data_list)

            if split in [self.SAMPLE_SPLIT, self.FEWSHOT_SPLIT]:
                self.dataset[split] = data_list

    def _get_initial_prompt_text(self, item: dict[str, Any]) -> str:
        return f"The following are multiple choice questions (with answers) about {item['subject']}."

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        instruction_text = item["question"].strip() + "\n"
        instruction_text += "".join([f"{key}. {choice}\n" for key, choice in zip(self.keys, item["options"])])
        return instruction_text

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        ground_truth = self._get_ground_truth(item)
        assert ground_truth is not None
        return f"{self._get_cue_text(item)}{ground_truth}"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        return f" {self.keys[item['answer_index']]}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [f" {key}" for key in self.keys]


class MMLU_PRO_DCS(MMLU_PRO):
    NAME = "MMLU Pro_DCS"
    METRICS = [
        AccuracyLoglikelihood,
        AccuracyNormLoglikelihood,
        ConfidenceWeightedAccuracy,
        DistributionalCorrectnessScore,
        TernaryScore
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


class MMLU_PRO_COT(MMLU_PRO):
    NAME = "MMLU_PRO_COT"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [AccuracyCompletion]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question", "Therefore", "the", "answer", "is", "ANSWER_LETTER"] + get_n_letters(
        4
    )
    ANS_RE = re.compile(r"Therefore, the answer is \(([ABCDEFGHIJ])\)")

    def __init__(self, num_fewshot: int = 0) -> None:
        assert num_fewshot == 0, "Fewshot is not supported for MMLU_PRO_COT"
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
        return self.keys[item["answer_index"]]

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        for stop_sequence in self.stop_sequences:
            if stop_sequence in completion_text:
                completion_text = completion_text.split(stop_sequence)[0]
        return self._extract_answer(completion_text)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        # using the reasoning prompt from "Figure 44 of Tülu 3 paper: https://arxiv.org/pdf/2411.15124"
        instruction_text = (
            "Answer the following multiple-choice question by giving the correct answer letter in parentheses. "
            "Provide CONCISE reasoning for the answer, and make sure to finish the response with "
            '"Therefore, the answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B), (C), (D), (E), etc.'
        )
        instruction_text += f"\n\nQuestion: {item['question'].strip()}\n"
        instruction_text += "\n".join([f"({key}) {choice}" for key, choice in zip(self.keys, item["options"])])
        instruction_text += (
            "\n\nAnswer the above question and REMEMBER to finish your response with the exact phrase "
            '"Therefore, the answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B), (C), (D), (E), etc.'
        )
        return instruction_text
