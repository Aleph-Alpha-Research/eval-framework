import hashlib
import logging
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
from eval_framework.tasks.base import NO_SUBJECT, RANDOM_SEED, BaseTask, Language, ResponseType, Sample, SubjectType
from eval_framework.tasks.utils import get_n_letters

logger = logging.getLogger(__name__)


class GPQA(BaseTask[str]):
    """GPQA dataset: https://huggingface.co/datasets/Idavidrein/gpqa"""

    NAME = "GPQA"
    DATASET_PATH = "Idavidrein/gpqa"
    SAMPLE_SPLIT = "train"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
    SUBJECTS = ["gpqa_extended"]  # ["gpqa_diamond", "gpqa_extended", "gpqa_main", "gpqa_experts"]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question"] + get_n_letters(4)
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        self.stop_sequences = ["Question:"]
        self.keys = get_n_letters(4)
        self.num_to_letter = {str(i): letter for i, letter in enumerate(self.keys, start=1)}
        self.rnd_choice_shuffle = random.Random(RANDOM_SEED)

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
                # exclude in the GPQA dataset one of the sample that has an too long prompt (DNA sequence)
                data_list_filtered = [
                    item
                    for item in data_list
                    if item["Question"]
                    != "Hello, you are embarking on a new project. You need to produce the HP1alpha protein in E. coli. Which of these plasmids will you choose?"  # noqa: E501
                ]
                if len(data_list) - len(data_list_filtered) > 0:
                    logger.info(f"Excluded {len(data_list) - len(data_list_filtered)} samples from {split} split.")
                assert len(data_list) - len(data_list_filtered) < 2, "we expect to remove max one item"

                self.dataset[split] = data_list_filtered

    def _get_initial_prompt_text(self, item: dict[str, Any]) -> str:
        system_prompt_text = (
            "Here are some example questions from experts. "
            "An explanation is given before the final answer. "
            "Answer the final question yourself, giving your reasoning beforehand."
        )
        return system_prompt_text

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        choices, _ = self._get_possible_completions_marked(item)
        prompt = f"Question: {item['Question'].strip()}\n"
        prompt += "\n".join(choices) + "\n"
        return prompt

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        ground_truth = self._get_ground_truth(item)
        assert ground_truth is not None
        return f"{self._get_cue_text(item)}{ground_truth}"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        choices, correct_answer_position = self._get_possible_completions_marked(item)
        answer_key = choices[correct_answer_position][:3]
        return f" {answer_key}"

    def _get_possible_completions_marked(self, item: dict[str, Any]) -> tuple[list[str], int]:
        choices = [self._preprocess(item[f"Incorrect Answer {x}"]) for x in range(1, 4)]
        correct_answer = self._preprocess(item["Correct Answer"])
        # we want to be random, but always the same for the same input
        # so we hash the string, which always give you the same seed
        hash_object = hashlib.sha256(f"{choices} {correct_answer}".encode())
        self.rnd_choice_shuffle.seed(int(hash_object.hexdigest(), 16))
        self.rnd_choice_shuffle.shuffle(choices)
        correct_answer_position = self.rnd_choice_shuffle.randint(0, 3)
        choices.insert(correct_answer_position, correct_answer)
        choices = [f"({self.keys[i]}) {choice}" for i, choice in enumerate(choices)]
        return choices, correct_answer_position

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [f" ({x})" for x in self.keys]

    @staticmethod
    def _preprocess(text: str | None) -> str:
        if text is None:
            return " "
        text = text.strip()
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text


class GPQA_DCS(GPQA):
    NAME = "GPQA_DCS"
    METRICS = [
        AccuracyLoglikelihood,
        AccuracyNormLoglikelihood,
        ConfidenceWeightedAccuracy,
        DistributionalCorrectnessScore,
        TernaryScore
    ]

    def _get_initial_prompt_text(self, item: dict[str, Any]) -> str:
        return (
            "Answer only if you are confident, since mistakes may be penalised, while correct answers receive points. "
            "It is acceptable to answer with '?' if you are unsure, and you will receive 0 points."
        )

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        completions = super()._get_possible_completions(item)
        return (completions or []) + [" (?)"]


class GPQA_COT(GPQA):
    NAME = "GPQA_COT"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [AccuracyCompletion]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question", "Therefore", "the", "answer", "is", "ANSWER_LETTER"] + get_n_letters(
        4
    )
    ANS_RE = re.compile(r"Therefore, the answer is \(([ABCDEFGHIJ])\)")

    def __init__(self, num_fewshot: int = 0) -> None:
        assert num_fewshot == 0, "Fewshot is not supported for GPQA_COT"
        super().__init__(num_fewshot)
        self.stop_sequences: list[str] = ["Question:"]
        self.keys = get_n_letters(4)
        self.num_to_letter = {str(i): letter for i, letter in enumerate(self.keys, start=1)}
        self.rnd_choice_shuffle = random.Random(RANDOM_SEED)

    def _extract_answer(self, completion: str) -> str:
        match = self.ANS_RE.search(completion)
        if match:
            match_str = match.group(1)
            return match_str
        else:
            return "[invalid]"

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        for stop_sequence in self.stop_sequences:
            if stop_sequence in completion_text:
                completion_text = completion_text.split(stop_sequence)[0]
        return self._extract_answer(completion_text)

    def _get_initial_prompt_text(self, item: dict[str, Any]) -> str:
        return ""

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        # using the reasoning prompt from "Figure 44 of Tülu 3 paper: https://arxiv.org/pdf/2411.15124"
        choices, _ = self._get_possible_completions_marked(item)
        instruction_text = (
            "Answer the following multiple-choice question by giving the correct answer letter in parentheses. "
            "Provide CONCISE reasoning for the answer, and make sure to finish the response with "
            '"Therefore, the answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B), (C), (D), (E), etc.'
        )
        instruction_text += f"\n\nQuestion: {item['Question'].strip()}\n"
        instruction_text += "\n".join(choices)
        instruction_text += (
            "\n\nAnswer the above question and REMEMBER to finish your response with the exact phrase "
            '"Therefore, the answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B), (C), (D), (E), etc.'
        )
        return instruction_text

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return ""

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        choices, correct_answer_position = self._get_possible_completions_marked(item)
        # index 1 selects the letter
        answer_key = choices[correct_answer_position][1]
        return answer_key
