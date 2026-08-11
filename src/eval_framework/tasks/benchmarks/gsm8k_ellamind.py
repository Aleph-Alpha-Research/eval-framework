"""German GSM8K (EllaMind) tasks.

https://huggingface.co/datasets/ellamind/gsm8k-platinum-multilingual
"""

import re
from typing import Any

from eval_framework.tasks.base import BaseTask, Language, Sample
from eval_framework.tasks.benchmarks.gsm8k import GSM8KEvalHarness
from eval_framework.tasks.dataset_revisions import HF_REVISIONS_LOCKFILE
from eval_framework.tasks.task_style import BPBStyle


class GSM8K_Ellamind_DE_Platinum(GSM8KEvalHarness):
    """German GSM8K (Platinum) generative, mirroring the English GSM8K.

    Dataset: https://huggingface.co/datasets/ellamind/gsm8k-platinum-multilingual

    German adaptations vs. the English ``GSM8K``:
      * prompt markers translated: ``Frage:`` / ``Antwort:``.
      * few-shot targets end with a German final-answer line
        (``Daher ist die Antwort N.``). The EllaMind ``final_answer`` column is the
        ground truth directly, so no ``#### N`` extraction is needed.
    """

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "GSM8K_Ellamind_DE_Platinum"
    DATASET_PATH = "ellamind/gsm8k-platinum-multilingual"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "test"
    SUBJECTS = ["deu"]
    LANGUAGE = Language.DEU

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        self.stop_sequences = ["Frage:"]

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"Frage: {item['question']}\n"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Antwort:"

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        return f"Antwort: {item['solution']} Daher ist die Antwort {self._get_ground_truth(item)}."

    def _get_ground_truth(self, item: dict[str, Any]) -> str:
        # Also remove possible thousand separator "." or "," from the ground truth.
        return self._normalize_number(item["final_answer"])

    def _normalize_number(self, answer: str) -> str:
        # Remove possible thousand separator "." or ","
        return answer.replace(".", "").replace(",", "")

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        """Extract the last number from the completion text."""
        # The EllaMind answers are integers, so extract all integers in the completion...)
        # Note, extraction will break for floats (e.g. "1.2" -> [12]) but no floats are expected.
        numbers = re.findall(r"[-+]?\d+", self._normalize_number(completion_text))
        # ... and return the last one.
        return numbers[-1] if numbers else "[invalid]"


class GSM8K_Ellamind_DE_BPB_Platinum(BaseTask):
    """German GSM8K (Platinum) BPB format."""

    REVISION_LOCKFILE = HF_REVISIONS_LOCKFILE

    NAME = "GSM8K_Ellamind_DE_BPB_Platinum"
    DATASET_PATH = "ellamind/gsm8k-platinum-multilingual"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "test"
    SUBJECTS = ["deu"]
    LANGUAGE = Language.DEU
    TASK_STYLER = BPBStyle(question_prefix="Frage: ", cue_text="Antwort:")

    def _get_raw_question(self, item: dict[str, Any]) -> str:
        return item["question"]

    def _get_choices(self, item: dict[str, Any]) -> list[str]:
        answer = f"{item['solution']} Daher ist die Antwort {item['final_answer']}."
        return [answer]

    def _get_correct_index(self, item: dict[str, Any]) -> int:
        return 0
