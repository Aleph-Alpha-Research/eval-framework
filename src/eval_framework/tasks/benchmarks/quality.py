import random
from typing import Any

from eval_framework.metrics.loglikelihood_metrics.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.tasks.base import RANDOM_SEED, BaseTask, Language, ResponseType, SubjectType
from eval_framework.tasks.dataloader import Dataloader


class QUALITY(BaseTask[str]):
    NAME = "QuALITY"
    DATASET_PATH = "emozilla/quality"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "validation"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
    SUBJECTS = ["hard", "easy"]

    PERTURBATION_UNMODIFIABLE_WORDS = ["Article", "Question", "Answer"]
    LANGUAGE = Language.ENG

    def __init__(self, dataloader: Dataloader, num_fewshot: int = 0) -> None:
        assert num_fewshot == 0, "QuALITY only supports zero fewshot examples"
        super().__init__(num_fewshot=num_fewshot, dataloader=dataloader)

    def _load_dataset(self, subject: SubjectType) -> None:
        hf_dataset = self.dataloader.load(path=self.DATASET_PATH)
        self.dataset = {}

        self.rnd = random.Random(RANDOM_SEED)

        for split, data in hf_dataset.items():
            if split not in [self.SAMPLE_SPLIT, self.FEWSHOT_SPLIT]:
                continue

            data_list = [item for item in data if item["hard"] == (subject == "hard")]

            if split == self.SAMPLE_SPLIT:
                self.rnd.shuffle(data_list)

            self.dataset[split] = data_list

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        article = item["article"]
        question = item["question"]
        return f"Article: {article}\nQuestion: {question}\n"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        return f" {item['options'][item['answer']]}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [f" {option}" for option in item["options"]]
