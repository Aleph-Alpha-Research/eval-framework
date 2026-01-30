import re
from typing import Any

from eval_framework.metrics.completion.exponential_similarity import ExponentialSimilarity
from eval_framework.metrics.completion.f1 import F1
from eval_framework.metrics.completion.rouge_geometric_mean import ROUGE_GEOMETRIC_MEAN
from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
)
from eval_framework.tasks.base import BaseTask, Language, ResponseType, Sample
from eval_framework.tasks.utils import get_n_letters


class ZERO_SCROLLS_QUALITY(BaseTask[str]):
    """ZeroSCROLLS dataset: https://huggingface.co/datasets/tau/zero_scrolls"""

    NAME = "ZeroSCROLLS QuALITY"
    DATASET_PATH = "tau/zero_scrolls"
    HF_REVISION = "3ee203cfad81b1a4fab8f2351c12679fbe95b179"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "validation"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood]
    SUBJECTS = ["quality"]

    PERTURBATION_UNMODIFIABLE_WORDS = ["Answer"]
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int = 0) -> None:
        assert num_fewshot == 0, "ZeroSCROLLS QuALITY only supports zero fewshot examples"
        super().__init__(num_fewshot)
        self.keys = get_n_letters(4)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        query_end_index = item["query_end_index"]
        return f"{item['input'][:query_end_index]}\n\n"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        return f" {item['output']}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [f" {key}" for key in self.keys]


class ZERO_SCROLLS_COMPLETION(BaseTask[str]):
    """ZeroSCROLLS dataset: https://huggingface.co/datasets/tau/zero_scrolls"""

    DATASET_PATH = "tau/zero_scrolls"
    HF_REVISION = "3ee203cfad81b1a4fab8f2351c12679fbe95b179"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "validation"
    RESPONSE_TYPE = ResponseType.COMPLETION

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        return item["output"]


class ZERO_SCROLLS_GOV_REPORT(ZERO_SCROLLS_COMPLETION):
    NAME = "ZeroSCROLLS GovReport"
    METRICS = [ROUGE_GEOMETRIC_MEAN]
    SUBJECTS = ["gov_report"]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Summary"]

    def __init__(self, num_fewshot: int = 0) -> None:
        assert num_fewshot == 0, "ZeroSCROLLS GovReport only supports zero fewshot examples"
        super().__init__(num_fewshot)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        query_end_index = item["query_end_index"]
        return f"{item['input'][:query_end_index]}Summary:"


class ZERO_SCROLLS_QMSUM(ZERO_SCROLLS_COMPLETION):
    NAME = "ZeroSCROLLS QMSum"
    METRICS = [ROUGE_GEOMETRIC_MEAN]
    SUBJECTS = ["qmsum"]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Answer"]

    def __init__(self, num_fewshot: int = 0) -> None:
        assert num_fewshot == 0, "ZeroSCROLLS QMSum only supports zero fewshot examples"
        super().__init__(num_fewshot)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        query_end_index = item["query_end_index"]
        return f"{item['input'][:query_end_index]}\n\nAnswer:"


class ZERO_SCROLLS_SQUALITY(ZERO_SCROLLS_COMPLETION):
    NAME = "ZeroSCROLLS SQuALITY"
    METRICS = [ROUGE_GEOMETRIC_MEAN]
    SUBJECTS = ["squality"]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Answer"]

    def __init__(self, num_fewshot: int = 0) -> None:
        assert num_fewshot == 0, "ZeroSCROLLS SQuALITY only supports zero fewshot examples"
        super().__init__(num_fewshot)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        query_end_index = item["query_end_index"]
        return f"{item['input'][:query_end_index]}\n\nAnswer:"


class ZERO_SCROLLS_QASPER(ZERO_SCROLLS_COMPLETION):
    NAME = "ZeroSCROLLS Qasper"
    METRICS = [F1]
    SUBJECTS = ["qasper"]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Answer"]

    def __init__(self, num_fewshot: int = 0) -> None:
        assert num_fewshot == 0, "ZeroSCROLLS Qasper only supports zero fewshot examples"
        super().__init__(num_fewshot)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        query_end_index = item["query_end_index"]
        return f"{item['input'][:query_end_index]}\n\nAnswer:"


class ZERO_SCROLLS_NARRATIVEQA(ZERO_SCROLLS_COMPLETION):
    NAME = "ZeroSCROLLS NarrativeQA"
    METRICS = [F1]
    SUBJECTS = ["narrative_qa"]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Answer"]

    def __init__(self, num_fewshot: int = 0) -> None:
        assert num_fewshot == 0, "ZeroSCROLLS NarrativeQA only supports zero fewshot examples"
        super().__init__(num_fewshot)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        query_end_index = item["query_end_index"]
        return f"{item['input'][:query_end_index]}\n\nAnswer:"


class ZERO_SCROLLS_MUSIQUE(ZERO_SCROLLS_COMPLETION):
    NAME = "ZeroSCROLLS MuSiQue"
    METRICS = [F1]
    SUBJECTS = ["musique"]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Answer"]

    def __init__(self, num_fewshot: int = 0) -> None:
        assert num_fewshot == 0, "ZeroSCROLLS MuSiQue only supports zero fewshot examples"
        super().__init__(num_fewshot)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        query_end_index = item["query_end_index"]
        return f"{item['input'][:query_end_index]}\n\nAnswer:"


class ZERO_SCROLLS_SPACE_DIGEST(ZERO_SCROLLS_COMPLETION):
    NAME = "ZeroSCROLLS SpaceDigest"
    METRICS = [ExponentialSimilarity]
    SUBJECTS = ["space_digest"]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Answer"]

    def __init__(self, num_fewshot: int = 0) -> None:
        assert num_fewshot == 0, "ZeroSCROLLS SpaceDigest only supports zero fewshot examples"
        super().__init__(num_fewshot)

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        # First, try to find patterns like "X%" or "X percent" or "X percentage"
        percentage_patterns = [
            r"(\d+(?:\.\d+)?)%",  # Matches: 30%, 30.5%
            r"(\d+(?:\.\d+)?)\s*percent",  # Matches: 30 percent, 30.5 percent
            r"(\d+(?:\.\d+)?)\s*percentage",  # Matches: 30 percentage, 30.5 percentage
            r"percentage\s*(?:is|of|:)?\s*(\d+(?:\.\d+)?)",  # Matches: percentage is 30, percentage: 30.5
            r"(?:is|equals|equal to|about|approximately|around|roughly)\s*(\d+(?:\.\d+)?)\s*%",
            # Matches: is 30%, equals 30.5%
            r"(?:is|equals|equal to|about|approximately|around|roughly)\s*(\d+(?:\.\d+)?)\s*percent",
            # Matches: is 30 percent
            r"it'?s\s*(\d+(?:\.\d+)?)",  # Matches: it's 60, its 60
            r"that'?s\s*(\d+(?:\.\d+)?)",  # Matches: that's 60, thats 60
        ]

        for pattern in percentage_patterns:
            match = re.search(pattern, completion_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # If no percentage pattern is found, check if the entire text is just a number
        if re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", completion_text):
            return completion_text.strip()

        # If not a standalone number, look for any number in the text
        # This is a fallback and might be less accurate
        number_match = re.search(r"(\d+(?:\.\d+)?)", completion_text)
        if number_match:
            return number_match.group(1).strip()

        # If no number is found, return the original text stripped
        return completion_text.strip()

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        query_end_index = item["query_end_index"]
        return f"{item['input'][:query_end_index]}Answer:"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        return self.post_process_generated_completion(item["output"])
