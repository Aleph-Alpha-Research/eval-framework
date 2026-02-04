from typing import Any

from eval_framework.metrics.completion.accuracy_completion import AccuracyCompletion
from eval_framework.metrics.completion.f1 import F1
from eval_framework.tasks.base import NO_SUBJECT, BaseTask, Language, ResponseType, Sample
from eval_framework.metrics.loglikelihood.bits_per_byte import BitsPerByteLoglikelihood


class SimpleQACompletion(BaseTask[str]):
    NAME = "SimpleQACompletion"
    DATASET_PATH = "lighteval/SimpleQA"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "few_shot"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [AccuracyCompletion, F1]
    SUBJECTS = [NO_SUBJECT]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question", "Answer"]
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        self.stop_sequences = ["Question:", "Answer:"]
        self.max_tokens = 50

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"Question: {item.get('problem', '')}\n"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_ground_truth(self, item: dict[str, Any]) -> str:
        return f" {item.get('answer', '')}"

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        gt = self._get_ground_truth(item)
        return f" {gt[0]}" if gt else ""

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        return completion_text.strip()