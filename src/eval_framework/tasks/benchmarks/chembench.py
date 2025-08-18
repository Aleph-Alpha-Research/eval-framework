import json
from typing import Any

from eval_framework.metrics.loglikelihood_metrics.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.tasks.base import BaseTask, Language, ResponseType
from eval_framework.tasks.utils import get_n_letters

CHEMBENCH_SUBJECTS = [
    "analytical_chemistry",
    "chemical_preference",
    "general_chemistry",
    "inorganic_chemistry",
    "materials_science",
    "organic_chemistry",
    "physical_chemistry",
    "technical_chemistry",
    "toxicity_and_safety",
]


class ChemBenchMultipleChoice(BaseTask[str]):
    """ChemBench dataset: https://huggingface.co/datasets/jablonkagroup/ChemBench"""

    NAME = "ChemBench"
    DATASET_PATH = "jablonkagroup/ChemBench"
    SAMPLE_SPLIT = "train"  # Only has train split
    FEWSHOT_SPLIT = "train"  # Only has train split
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
    SUBJECTS = CHEMBENCH_SUBJECTS
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int = 0) -> None:
        assert num_fewshot == 0, "Fewshot is not supported for ChemBench"
        super().__init__(num_fewshot)

        self.keys = get_n_letters(16)

    def _load_dataset(self, subject: str) -> None:
        super()._load_dataset(subject)
        # Keep only the multiple-choice options with 1 correct answer
        for split in self.dataset.keys():
            filtered_items = []
            for item in self.dataset[split]:
                if item.get("metrics") == ["multiple_choice_grade"]:
                    target_scores = json.loads(item["examples"][0]["target_scores"])
                    correct_answers = [i for i, score in enumerate(target_scores.values()) if score == 1.0]
                    if len(correct_answers) == 1:
                        filtered_items.append(item)
            self.dataset[split] = filtered_items

    def _get_subject_name(self, item: dict[str, Any]) -> str:
        return " ".join(item["subject"].split("_"))

    def _get_initial_prompt_text(self, item: dict[str, Any]) -> str:
        return (
            "The following is a question about chemistry. Please answer by responding with the letter of the correct "
            "answer."
        )

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        question = item["examples"][0]["input"].strip()
        target_scores = json.loads(item["examples"][0]["target_scores"])
        choices = "".join([f"{key}. {choice}\n" for key, choice in zip(self.keys, target_scores.keys())])
        return f"Question: {question}\n{choices}"

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        ground_truth = self._get_ground_truth(item)
        return f"{self._get_cue_text(item)}{ground_truth}"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        target_scores = json.loads(item["examples"][0]["target_scores"])
        correct_answers = [i for i, score in enumerate(target_scores.values()) if score == 1.0]
        assert len(correct_answers) == 1, f"Expected exactly one correct answer, but got {len(correct_answers)}"
        return f" {self.keys[correct_answers[0]]}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        target_scores = json.loads(item["examples"][0]["target_scores"])
        return [f" {key}" for key in self.keys[: len(target_scores)]]
