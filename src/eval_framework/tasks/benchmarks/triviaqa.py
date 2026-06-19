import random
from typing import Any

from eval_framework.metrics.completion.accuracy_completion import AccuracyCompletion
from eval_framework.metrics.completion.f1 import F1, F1SquadNormalized
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

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
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


class TriviaQAMA(TRIVIAQA):
    """TriviaQA with the exact system prompt used in MA training"""

    NAME = "TriviaQA_MA"
    SUBJECTS = ["rc.wikipedia"]
    UNANSWERABLE_STR = "unanswerable"

    METRICS = [AccuracyCompletion, F1, F1SquadNormalized]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question", "Answer", "Context", "unanswerable"]

    def _get_context_text(self, item: dict[str, Any]) -> str:
        return "\n\n".join(item["entity_pages"]["wiki_context"])

    def _get_system_prompt_text(self, item: dict[str, Any]) -> str | None:
        return (
            "You are a helpful assistant and will answer the user's questions carefully, "
            "logically, accurately and well-reasoned.\n"
            "Use the given context to answer the question faithfully. Answer only if the "
            f"answer is present in the given context, otherwise respond with '{self.UNANSWERABLE_STR}' "
            "if the answer is not present in the context."
        )

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"Context:\n{self._get_context_text(item)}\n\nQuestion:\n{item['question'].strip()}\n"
