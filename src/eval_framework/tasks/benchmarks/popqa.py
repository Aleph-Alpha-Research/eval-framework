import json
from typing import Any

from eval_framework.metrics.completion.accuracy_completion import AccuracyCompletion
from eval_framework.tasks.base import NO_SUBJECT, BaseTask, Language, ResponseType, Sample


class PopQA(BaseTask[str]):
    NAME = "PopQA"
    DATASET_PATH = "akariasai/PopQA"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "test"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [AccuracyCompletion]
    SUBJECTS = [NO_SUBJECT]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question"]
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        self.stop_sequences = ["Question:", "\n\n"]
        self.max_tokens = 15

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"Question: {item.get('question', '')}\n"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_ground_truth(self, item: dict[str, Any]) -> list[str]:
        return [f" {a}" for a in item.get("possible_answers", [])]

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        return completion_text.strip().lower()
