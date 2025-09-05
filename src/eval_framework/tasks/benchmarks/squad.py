import random
from typing import Any

from eval_framework.metrics.completion.accuracy_completion import AccuracyCompletion
from eval_framework.metrics.completion.f1 import F1
from eval_framework.tasks.base import NO_SUBJECT, RANDOM_SEED, BaseTask, Language, ResponseType, SubjectType


class SQUAD2(BaseTask[str]):
    """Squad v2 dataset: https://huggingface.co/datasets/rajpurkar/squad_v2"""

    NAME = "SQuAD2"
    DATASET_PATH = "rajpurkar/squad_v2"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [AccuracyCompletion, F1]
    SUBJECTS = [NO_SUBJECT]
    UNANSWERABLE_STR = "unanswerable"
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question", "Answer", "Context", "unanswerable"]
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int, custom_subjects: list[str] | None, custom_hf_revision: str | None) -> None:
        super().__init__(num_fewshot, custom_subjects, custom_hf_revision)
        self.stop_sequences = [".\n"]
        self.max_tokens = 300  # the max length of the ground truth is 160 characters while the average is ~19
        self.rnd_choice_shuffle = random.Random()

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
                self.dataset[split] = data_list

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        prompt = (
            "Given the following context, answer the question. If the question cannot be answered based "
            f"on the context alone, respond with '{self.UNANSWERABLE_STR}'.\n\n"
            "Context:\n"
            f"{item['context']}\n\n"
            f"Question:\n{item['question']}\nAnswer:"
        )
        return prompt

    def _get_ground_truth(self, item: dict[str, Any]) -> list[str]:
        text_ = item["answers"]["text"]
        ground_truth_for_unanswerable = [
            self.UNANSWERABLE_STR,
            self.UNANSWERABLE_STR + " ",
            self.UNANSWERABLE_STR.capitalize(),
        ]
        ground_truths = text_ if text_ else ground_truth_for_unanswerable
        return [f" {ground_truth}" for ground_truth in ground_truths]

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        target = self._get_ground_truth(item)[0]
        assert target is not None
        assert isinstance(target, str)
        return target


class SQUAD(SQUAD2):
    """Squad dataset: https://huggingface.co/datasets/rajpurkar/squad"""

    NAME = "SQuAD"
    DATASET_PATH = "rajpurkar/squad"

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        prompt = (
            "Given the following context, answer the question.\n\n"
            "Context:\n"
            f"{item['context']}\n\n"
            f"Question:\n{item['question']}\n"
        )
        return prompt

    def _get_ground_truth(self, item: dict[str, Any]) -> list[str]:
        return item["answers"]["text"]
