import random
from typing import Any

from eval_framework.metrics.completion.accuracy_completion import AccuracyCompletion
from eval_framework.metrics.completion.f1 import F1
from eval_framework.tasks.base import BaseTask, Language, ResponseType, Sample


class TRIVIAQA(BaseTask[str]):
    """Trivia QA dataset: https://huggingface.co/datasets/mandarjoshi/trivia_qa"""

    NAME = "TriviaQA"
    DATASET_PATH = "mandarjoshi/trivia_qa"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [AccuracyCompletion, F1]
    SUBJECTS = ["rc.wikipedia.nocontext"]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question", "Answer"]
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int, custom_subjects: list[str] | None, custom_hf_revision: str | None) -> None:
        super().__init__(num_fewshot, custom_subjects, custom_hf_revision)
        self.stop_sequences = ["\n"]
        self.max_tokens = 400  # the max length of the ground truth is 282 characters while the average is ~16
        self.rnd_choice_shuffle = random.Random()

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        prompt = f"Question: {item['question'].strip()}\nAnswer:"
        return prompt

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        target = self._get_ground_truth(item)[0]
        assert target is not None
        assert isinstance(target, str)
        return f" {target}"

    def _get_ground_truth(self, item: dict[str, Any]) -> list[str]:
        return item["answer"]["aliases"]

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        return completion_text.strip().rstrip(".")
