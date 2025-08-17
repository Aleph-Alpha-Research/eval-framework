from typing import Any

from eval_framework.metrics.loglikelihood_metrics.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.tasks.base import BaseTask, Language, ResponseType
from eval_framework.tasks.utils import get_n_letters

INCLUDE_SUBJECTS = [
    "Albanian",
    "Arabic",
    "Armenian",
    "Azerbaijani",
    "Basque",
    "Belarusian",
    "Bengali",
    "Bulgarian",
    "Chinese",
    "Croatian",
    "Dutch",
    "Estonian",
    "Finnish",
    "French",
    "Georgian",
    "German",
    "Greek",
    "Hebrew",
    "Hindi",
    "Hungarian",
    "Indonesian",
    "Italian",
    "Japanese",
    "Kazakh",
    "Korean",
    "Lithuanian",
    "Malay",
    "Malayalam",
    "Nepali",
    "North Macedonian",
    "Persian",
    "Polish",
    "Portuguese",
    "Russian",
    "Serbian",
    "Spanish",
    "Tagalog",
    "Tamil",
    "Telugu",
    "Turkish",
    "Ukrainian",
    "Urdu",
    "Uzbek",
    "Vietnamese",
]


def subject_to_language(subject: str) -> Language:
    if subject == "Greek":
        return Language.ELL  # type: ignore[attr-defined]
    elif subject == "Malay":
        return Language.MSA  # type: ignore[attr-defined]
    elif subject == "Nepali":
        return Language.NEP  # type: ignore[attr-defined]
    elif subject == "North Macedonian":
        return Language.MKD  # type: ignore[attr-defined]
    else:
        return Language(subject)


class INCLUDE(BaseTask[str]):
    """INCLUDE dataset: https://huggingface.co/datasets/CohereLabs/include-base-44"""

    NAME = "INCLUDE"
    DATASET_PATH = "CohereLabs/include-base-44"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "validation"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
    SUBJECTS = INCLUDE_SUBJECTS
    LANGUAGE = {lang: subject_to_language(lang) for lang in INCLUDE_SUBJECTS}

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        self.keys = get_n_letters(4)

    def _get_initial_prompt_text(self, item: dict[str, Any]) -> str:
        return f"The following are multiple choice questions (with answers) in {item['language']}."  # noqa: E501

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        question = item["question"].strip()
        choices = "".join(
            [
                f"{key}. {choice}\n"
                for key, choice in zip(
                    self.keys, [item["option_a"], item["option_b"], item["option_c"], item["option_d"]]
                )
            ]
        )
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
